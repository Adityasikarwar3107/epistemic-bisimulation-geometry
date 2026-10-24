"""Merged manuscript tables; constant columns are asserted and moved to the captions."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"
OUT = RES/"compact"
BS = chr(92)


def load(path, key):
    return json.loads((RES/path).read_text())[key]


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text("\n".join(lines)+"\n"+BS+"bottomrule\n", encoding="utf8")


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def cell(rows, **key):
    for r in rows:
        if all(r.get(k) == v for k, v in key.items()):
            return r
    return None


def v1_combined():
    """V1 calibration, V5 baselines and V8 refinements on the same V1 samples."""
    v1 = load("verified/summary.json", "calibration")
    v5 = load("extended/summary_baselines.json", "baselines")
    v8 = load("extended2/summary_v1.json", "v1")
    lines = []
    for model, states, gamma in [("ring6_g0.6", 6, .6), ("ring8_g0.9", 8, .9)]:
        if lines:
            lines.append(BS+"midrule")
        for n in [0, 128, 2048, 8192, 131072]:
            base = cell(v1, model=model, n_per_row=n)
            assert base["coverage"] == 1.0
            row = [str(states), str(gamma), f"{n:,}", f3(base["mean_error"]), f3(base["mean_width"])]
            for method in ["global", "support"]:
                r = cell(v8, model=model, method=method, n_per_row=n)
                assert r["coverage"] == 1.0
                row.append(f3(r["mean_sup_width"]))
            sim = cell(v5, model=model, method="global", n_per_row=n)
            if sim is not None:
                assert sim["coverage"] == 1.0
            row.append(f3(sim and sim["mean_sup_width"]))
            for method in ["bootstrap_pointwise", "bootstrap_band"]:
                r = cell(v5, model=model, method=method, n_per_row=n)
                row += [f3(r and r["coverage"]), f3(r and r["mean_sup_width"])]
            lines.append(" & ".join(row)+BS*2)
    write("v1_combined_rows.tex", lines)


def v6_combined():
    """V6 scaling with the V8 refinements and bootstrap band at the final count."""
    v6 = load("extended/summary_scaling.json", "scaling")
    v8 = load("extended2/summary_v6.json", "v6")
    lines = []
    for states in [16, 32, 64, 128]:
        row = [str(states)]
        for n in [2048, 8192, 32768, 131072]:
            r = cell(v6, states=states, n_per_row=n)
            assert r["coverage"] == 1.0
            row.append(f3(r["mean_sup_width"]))
        for method in ["global", "support", "bootstrap_band"]:
            r = cell(v8, states=states, method=method, n_per_row=131072)
            if r is not None and method != "bootstrap_band":
                assert r["coverage"] == 1.0
            row.append(f3(r and r["mean_sup_width"]))
        final = cell(v6, states=states, n_per_row=131072)
        row += [f"{final['mean_anchors']:.2f}", f3(final["max_anchor_distortion"]),
                f"{final['median_certificate_seconds']:.2f}"]
        lines.append(" & ".join(row)+BS*2)
    write("v6_combined_rows.tex", lines)


def v3_compact():
    groups = load("verified/summary.json", "ablations")["groups"]
    human = {"corridor": "Corridor", "branching_return": "Branching return"}
    lines = []
    for task in ["corridor", "branching_return"]:
        for nav in ["random", "optimistic"]:
            row = [human[task], nav.capitalize()]
            for priority in ["local", "propagated"]:
                g = cell(groups, task=task, navigation=nav, priority=priority)
                assert g["completed"] == g["runs"] == 32
                row.append(f"{g['mean']:,.1f} [{g['ci95'][0]:,.1f}, {g['ci95'][1]:,.1f}]")
            lines.append(" & ".join(row)+BS*2)
    write("v3_rows.tex", lines)


def v7_compact():
    pre = load("extended/summary_anytime.json", "anytime")["groups"]
    post = load("extended/summary_anytime_posthoc.json", "anytime_posthoc")["groups"]
    human = {"corridor": "Corridor", "branching_return": "Branching"}
    lines = []
    for task in ["corridor", "branching_return"]:
        if lines:
            lines.append(BS+"midrule")
        for ties, groups in [("First action", pre), ("Unresolved row", post)]:
            for g in [r for r in groups if r["task"] == task]:
                assert g["completed"] == g["runs"] == 32
                s = g["steps_to_quota"]
                row = [human[task], g["priority"].capitalize(), ties,
                       f"{s['mean']:,.0f} [{s['ci95'][0]:,.0f}, {s['ci95'][1]:,.0f}]"]
                row += [f3(g[f"mean_width_at_{b}"]["mean"]) for b in [20000, 40000, 80000]]
                lines.append(" & ".join(row)+BS*2)
    write("v7_rows.tex", lines)


def riverswim_compact():
    cells = [c for c in load("extended2/summary_riverswim.json", "riverswim")["cells"] if "n_per_row" in c]
    lines = []
    for n in sorted({c["n_per_row"] for c in cells}):
        row = [f"{n:,}"]
        for method in ["baseline", "global", "support"]:
            c = cell(cells, method=method, n_per_row=n)
            assert c["coverage"] == 1.0
            row.append(f3(c["mean_sup_width"]))
        band = cell(cells, method="bootstrap_band", n_per_row=n)
        row += [f3(band["coverage"]), f3(band["mean_sup_width"])]
        lines.append(" & ".join(row)+BS*2)
    write("riverswim_rows.tex", lines)


def priorities_compact():
    groups = load("extended2/summary_priorities.json", "priorities")["groups"]
    human = {"corridor": "Corridor", "branching_return": "Branching", "riverswim": "RiverSwim"}
    lines = []
    for task in ["corridor", "branching_return", "riverswim"]:
        if lines:
            lines.append(BS+"midrule")
        for g in [r for r in groups if r["task"] == task]:
            assert g["completed"] == g["runs"] == 32
            keys = sorted([k for k in g if k.startswith("mean_width_at_")], key=lambda k: int(k.split("_")[-1]))
            row = [human[task], g["rule"].capitalize(), f"{g['steps']['mean']:,.0f}"]
            row += [f3(g[k]["mean"]) for k in keys[-3:]]
            lines.append(" & ".join(row)+BS*2)
    write("priority_rows.tex", lines)


def targeted_compact():
    groups = load("extended2/summary_targeted.json", "targeted")["groups"]
    human = {"corridor": "Corridor", "riverswim": "RiverSwim"}
    lines = []
    for g in groups:
        assert g["certified"] == g["runs"] == 32
        s = g["steps"]
        lines.append(" & ".join([human[g["task"]], "Adjoint" if g["rule"] == "adjoint_target" else "Count",
                                 f"{s['mean']:,.0f} [{s['ci95'][0]:,.0f}, {s['ci95'][1]:,.0f}]",
                                 f"{100*g['target_share']['mean']:.0f}"])+BS*2)
    write("targeted_rows.tex", lines)


if __name__ == "__main__":
    for part in [v1_combined, v6_combined, v3_compact, v7_compact, riverswim_compact,
                 priorities_compact, targeted_compact]:
        part()
    for p in sorted(OUT.glob("*.tex")):
        print("==", p.name)
        print(p.read_text())
