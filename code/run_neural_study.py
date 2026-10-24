"""V11: the neural extension on a block MDP with an exogenous distractor."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from finite_batch import solve
from run_verified_study import ring
from neural_primitives import SymmetricWidthCritic, surrogate_endpoints, representation_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results"/"neural"
SPEC = {
    "version": "neural-study-v1", "latent": "ring S=6, gamma=0.6 (V1)", "gamma": .6,
    "exogenous_states": 5, "exogenous_kernel": "stay .5, move +-1 mod 5 with .25 each",
    "observation_dim": 24, "signal_scale": 1., "distractor_scale": 1.5, "noise": .1,
    "seeds": [70000, 70001, 70002, 70003, 70004], "methods": ["interval", "point", "raw"],
    "epochs": 3000, "batch_pairs": 64, "successors": 8, "ensemble": 5, "beta_cal": 2.,
    "zeta": .05, "learning_rate": 1e-3, "ema": .01, "latent_dim": 8, "hidden": 64,
    "test_observations": 300, "test_pairs": 2000, "eval_epochs": [500, 1000, 2000, 3000],
    "scope": "Generative access to successor observations; surrogate intervals are not certificates",
}


class World:
    def __init__(self, seed):
        self.model = ring(6, SPEC["gamma"])
        self.metric = solve(self.model.reward, self.model.transition, SPEC["gamma"])["truth"][0]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
        d = SPEC["observation_dim"]
        self.signal = SPEC["signal_scale"]*rng.normal(size=(6, d))/np.sqrt(d)
        self.distractor = SPEC["distractor_scale"]*rng.normal(size=(SPEC["exogenous_states"], d))/np.sqrt(d)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 8]))

    def observe(self, s, x):
        noise = SPEC["noise"]*self.rng.normal(size=(len(s), SPEC["observation_dim"]))
        return self.signal[s]+self.distractor[x]+noise

    def sample_states(self, n):
        return self.rng.integers(6, size=n), self.rng.integers(SPEC["exogenous_states"], size=n)

    def step(self, s, x, a, m):
        """m successor latents, exogenous states, and Bernoulli rewards for each (s, x) under action a."""
        p = self.model.transition[s, a]
        u = self.rng.random((len(s), m, 1))
        s_next = (u > np.cumsum(p, -1)[:, None, :]).sum(-1)
        move = self.rng.choice([-1, 0, 0, 1], size=(len(s), m))
        x_next = (x[:, None]+move) % SPEC["exogenous_states"]
        reward = self.rng.random((len(s), m)) < self.model.reward[s, a][:, None]
        return s_next, x_next, reward.astype(float)


def mlp(inp, out):
    h = SPEC["hidden"]
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def ema(target, online, rate):
    with torch.no_grad():
        for t, o in zip(target.parameters(), online.parameters()):
            t.mul_(1-rate).add_(rate*o)


def assignment_ot(cost):
    """Exact transport between two uniform m-point sets; returns value and matched indices."""
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols].mean(), cols


def targets(world, encoder, critic, s, x, s2, x2, rng, with_width):
    """Ensemble point target M, proxy source b, and width targets for a batch of pairs."""
    gamma, m, k_heads = SPEC["gamma"], SPEC["successors"], SPEC["ensemble"]
    n = len(s)
    q = np.zeros((n, 2, k_heads))
    plans = []
    for a in range(2):
        sa, xa, ra = world.step(s, x, a, m)
        sb, xb, rb = world.step(s2, x2, a, m)
        with torch.no_grad():
            za = encoder(torch.tensor(world.observe(sa.ravel(), xa.ravel()), dtype=torch.float32)).numpy().reshape(n, m, -1)
            zb = encoder(torch.tensor(world.observe(sb.ravel(), xb.ravel()), dtype=torch.float32)).numpy().reshape(n, m, -1)
        cost = np.minimum(1., np.linalg.norm(za[:, :, None, :]-zb[:, None, :, :], axis=-1))
        matched = []
        for i in range(n):
            for k in range(k_heads):
                ia = rng.integers(m, size=m) if k else np.arange(m)
                ib = rng.integers(m, size=m) if k else np.arange(m)
                value, cols = assignment_ot(cost[i][np.ix_(ia, ib)])
                q[i, a, k] = (1-gamma)*abs(ra[i, ia].mean()-rb[i, ib].mean())+gamma*value
                if k == 0:
                    matched.append((za[i], zb[i][cols]))
        plans.append(matched)
    point = q.mean(-1).max(-1)
    source = SPEC["beta_cal"]*q.std(-1, ddof=1)
    out = {"point": point, "source": source}
    if with_width:
        width_target = np.zeros((n, 2))
        for a in range(2):
            u = torch.tensor(np.stack([p[0] for p in plans[a]]), dtype=torch.float32)
            v = torch.tensor(np.stack([p[1] for p in plans[a]]), dtype=torch.float32)
            with torch.no_grad():
                w_next = torch.stack([critic(u.reshape(-1, u.shape[-1]), v.reshape(-1, v.shape[-1]),
                                             torch.full((u.shape[0]*u.shape[1],), b, dtype=torch.long))
                                      for b in range(2)]).max(0).values.reshape(n, -1).mean(-1).numpy()
            width_target[:, a] = np.minimum(1., 2*source[:, a]+gamma*w_next)
        out["width_target"] = width_target
    return out


def evaluate(world, encoder, target_encoder, critic, test, rng, method):
    s, x, obs = test
    if encoder is not None:
        with torch.no_grad():
            z = encoder(torch.tensor(obs, dtype=torch.float32)).numpy()
    i, j = rng.choice(len(s), size=(2, SPEC["test_pairs"]))
    keep = i != j
    i, j = i[keep], j[keep]
    truth = world.metric[s[i], s[j]]
    if method == "raw":
        raw = np.linalg.norm(obs[i]-obs[j], axis=-1)
        scale = float(raw @ truth/(raw @ raw))
        dist = scale*raw
    else:
        dist = np.linalg.norm(z[i]-z[j], axis=-1)
    err = np.abs(dist-truth)
    same = s[i] == s[j]
    record = {"sup_distortion": float(err.max()), "mean_distortion": float(err.mean()),
              "same_latent_mean_distance": float(dist[same].mean()) if same.any() else float("nan"),
              "different_latent_mean_distance": float(dist[~same].mean())}
    if method == "interval":
        t = targets(world, target_encoder, critic, s[i], x[i], s[j], x[j], rng, False)
        with torch.no_grad():
            zi = target_encoder(torch.tensor(obs[i], dtype=torch.float32))
            zj = target_encoder(torch.tensor(obs[j], dtype=torch.float32))
            width = torch.stack([critic(zi, zj, torch.full((len(i),), b, dtype=torch.long))
                                 for b in range(2)]).max(0).values
        ends = surrogate_endpoints(torch.tensor(t["point"], dtype=torch.float32), width)
        lo, hi = ends["lower"].numpy(), ends["upper"].numpy()
        record.update(interval_coverage=float(((lo <= truth) & (truth <= hi)).mean()),
                      interval_mean_width=float((hi-lo).mean()))
    return record


def run(seed, method):
    torch.manual_seed(seed)
    world = World(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9]))
    ts, tx = world.sample_states(SPEC["test_observations"])
    test = (ts, tx, world.observe(ts, tx))
    records = []
    if method == "raw":
        return [{"seed": seed, "method": method, "epoch": 0, **evaluate(world, None, None, None, test, rng, method)}]
    d = SPEC["latent_dim"]
    encoder, target_encoder = mlp(SPEC["observation_dim"], d), mlp(SPEC["observation_dim"], d)
    target_encoder.load_state_dict(encoder.state_dict())
    critic, target_critic = SymmetricWidthCritic(d, 2, hidden=SPEC["hidden"]), SymmetricWidthCritic(d, 2, hidden=SPEC["hidden"])
    target_critic.load_state_dict(critic.state_dict())
    opt = torch.optim.Adam(encoder.parameters(), lr=SPEC["learning_rate"])
    copt = torch.optim.Adam(critic.parameters(), lr=SPEC["learning_rate"])
    for epoch in range(1, SPEC["epochs"]+1):
        s, x = world.sample_states(SPEC["batch_pairs"])
        s2, x2 = world.sample_states(SPEC["batch_pairs"])
        t = targets(world, target_encoder, target_critic, s, x, s2, x2, rng, method == "interval")
        o1 = torch.tensor(world.observe(s, x), dtype=torch.float32)
        o2 = torch.tensor(world.observe(s2, x2), dtype=torch.float32)
        point = torch.tensor(t["point"], dtype=torch.float32)
        if method == "interval":
            with torch.no_grad():
                z1, z2 = target_encoder(o1), target_encoder(o2)
            loss_c = sum(((critic(z1, z2, torch.full((len(s),), a, dtype=torch.long))
                           - torch.tensor(t["width_target"][:, a], dtype=torch.float32))**2).mean() for a in range(2))
            copt.zero_grad()
            loss_c.backward()
            copt.step()
            with torch.no_grad():
                width = torch.stack([target_critic(z1, z2, torch.full((len(s),), a, dtype=torch.long))
                                     for a in range(2)]).max(0).values
            ends = surrogate_endpoints(point, width)
            midpoint, interval = ends["midpoint"], ends["interval_width"]
        else:
            midpoint, interval = point, torch.zeros_like(point)
        dist = (encoder(o1)-encoder(o2)).norm(dim=-1)
        loss = representation_loss(dist, midpoint, interval, torch.zeros(len(s), dtype=torch.bool), zeta=SPEC["zeta"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema(target_encoder, encoder, SPEC["ema"])
        ema(target_critic, critic, SPEC["ema"])
        if epoch in SPEC["eval_epochs"]:
            rec = evaluate(world, encoder, target_encoder, target_critic, test, rng, method)
            records.append({"seed": seed, "method": method, "epoch": epoch, **rec})
            print(seed, method, epoch, {k: round(v, 3) for k, v in rec.items()}, flush=True)
    return records


def _job(job):
    torch.set_num_threads(1)
    return run(*job)


def main():
    from concurrent.futures import ProcessPoolExecutor
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT/"experiment_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != SPEC:
        raise RuntimeError("The frozen specification differs; use a new version/output folder")
    spec_path.write_text(json.dumps(SPEC, indent=2)+"\n", encoding="utf8")
    started = time.time()
    jobs = [(seed, method) for method in SPEC["methods"] for seed in SPEC["seeds"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for part in pool.map(_job, jobs) for r in part]
    with (OUT/"neural.csv").open("w", newline="", encoding="utf8") as stream:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    final = {}
    for method in SPEC["methods"]:
        last = [r for r in rows if r["method"] == method and r["epoch"] == max(rr["epoch"] for rr in rows if rr["method"] == method)]
        final[method] = {k: {"mean": float(np.mean([r[k] for r in last])), "min": float(np.min([r[k] for r in last])),
                             "max": float(np.max([r[k] for r in last]))}
                         for k in last[0] if k not in ["seed", "method", "epoch"]}
    summary = {"final": final, "elapsed_seconds": time.time()-started, "torch": torch.__version__}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
    manifest = {"spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT/"code/neural_primitives.py"]},
                "result_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [OUT/"neural.csv", OUT/"summary.json"]}}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
