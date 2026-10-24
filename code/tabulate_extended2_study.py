"""Tables and macros for V8-V11."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results"/"extended2"
NEURAL = ROOT/"results"/"neural"
MACROS = {}
BS = chr(92)


def load(name):
    path = DATA/f"summary_{name}.json"
    return json.loads(path.read_text())[name] if path.exists() else None


def write(name, lines):
    (DATA/name).write_text("\n".join(lines)+"\n"+BS+"bottomrule\n", encoding="utf8")


def fmt(x, digits=3):
    return "--" if x is None else f"{x:.{digits}f}"


def widths():
    """V8 table: mean supremum width by certificate, all with full coverage unless stated."""
    v1, v6, rs = load("v1"), load("v6"), load("riverswim")
    lines, coverage_failures = [], []

    def cell(rows, **key):
        for r in rows:
            if all(r.get(k) == v for k, v in key.items()) and "mean_sup_width" in r:
                return r
        return None
    if v1:
        for model, label in [("ring6_g0.6", "Ring, $S=6$, $\\gamma=0.6$"), ("ring8_g0.9", "Ring, $S=8$, $\\gamma=0.9$")]:
            for n in [8192, 131072]:
                row = [label, f"{n:,}"]
                for method in ["baseline", "global", "support"]:
                    c = cell(v1, model=model, method=method, n_per_row=n)
                    row.append(fmt(c["mean_sup_width"]))
                    if c["coverage"] < 1:
                        coverage_failures.append((model, method, n))
                v5 = json.loads((ROOT/"results"/"extended"/"summary_baselines.json").read_text())["baselines"]
                band = next(r for r in v5 if r.get("model") == model and r.get("method") == "bootstrap_band"
                            and r.get("n_per_row") == n)
                row.append(fmt(band["mean_sup_width"]))
                lines.append(" & ".join(row)+BS*2)
        for model, tag in [("ring6_g0.6", "Low"), ("ring8_g0.9", "High")]:
            base = cell(v1, model=model, method="baseline", n_per_row=131072)["mean_sup_width"]
            sup = cell(v1, model=model, method="support", n_per_row=131072)["mean_sup_width"]
            glob = cell(v1, model=model, method="global", n_per_row=131072)["mean_sup_width"]
            MACROS[f"RefSupportRatio{tag}"] = f"{base/sup:.1f}"
            MACROS[f"RefGlobalRatio{tag}"] = f"{base/glob:.1f}"
            MACROS[f"RefSupportWidth{tag}"] = f"{sup:.3f}"
            whole = [r for r in v1 if r.get("model") == model and "whole_run_successes" in r]
            MACROS[f"RefWhole{tag}"] = str(min(r["whole_run_successes"] for r in whole))
            v5 = json.loads((ROOT/"results"/"extended"/"summary_baselines.json").read_text())["baselines"]
            band = next(r for r in v5 if r.get("model") == model and r.get("method") == "bootstrap_band"
                        and r.get("n_per_row") == 131072)["mean_sup_width"]
            MACROS[f"RefBandGapBase{tag}"] = f"{base/band:.0f}"
            MACROS[f"RefBandGapSupport{tag}"] = f"{sup/band:.0f}"
    if v6:
        lines.append(BS+"midrule")
        for states in [16, 32, 64, 128]:
            row = [f"Lumpable, $S={states}$", "131,072"]
            for method in ["baseline", "global", "support", "bootstrap_band"]:
                c = cell(v6, states=states, method=method, n_per_row=131072)
                row.append(fmt(c["mean_sup_width"]) if c else "--")
                if c and method != "bootstrap_band" and c["coverage"] < 1:
                    coverage_failures.append((states, method))
            lines.append(" & ".join(row)+BS*2)
        whole = [r for r in v6 if "whole_run_successes" in r and r["method"] != "bootstrap_band"]
        MACROS["RefVsixWhole"] = str(sum(r["whole_run_successes"] for r in whole))
        MACROS["RefVsixRuns"] = str(sum(r["runs"] for r in whole))
    if rs:
        lines.append(BS+"midrule")
        cells = rs["cells"]
        for n in [8192, 131072]:
            row = ["RiverSwim, $\\gamma=0.9$", f"{n:,}"]
            for method in ["baseline", "global", "support", "bootstrap_band"]:
                c = cell(cells, method=method, n_per_row=n)
                row.append(fmt(c["mean_sup_width"]))
                if method != "bootstrap_band" and c["coverage"] < 1:
                    coverage_failures.append(("riverswim", method, n))
            lines.append(" & ".join(row)+BS*2)
    write("refined_rows.tex", lines)
    MACROS["RefCoverageFailures"] = str(len(coverage_failures))
    return coverage_failures


def decisions():
    v6 = load("v6")
    if not v6:
        return
    lines = []
    for states in [16, 32]:
        for n in [8192, 32768, 131072]:
            row = [str(states), f"{n:,}"]
            for method in ["baseline", "support", "bootstrap_band"]:
                c = next(r for r in v6 if r.get("states") == states and r.get("method") == method and r.get("n_per_row") == n)
                row += [f"{100*c['zero_pairs_merged']:.0f}", f"{100*c['far_pairs_split']:.0f}",
                        str(c["runs_with_false_merge"]+c["runs_with_false_split"])]
            lines.append(" & ".join(row)+BS*2)
    write("decision_rows.tex", lines)
    band = [r for r in v6 if r.get("method") == "bootstrap_band" and "n_per_row" in r]
    MACROS["DecBandFalseRunsMax"] = str(max(r["runs_with_false_merge"]+r["runs_with_false_split"] for r in band))
    for states, word in [(16, "Small"), (32, "Large")]:
        cells = [r for r in band if r["states"] == states]
        wrong = [r["runs_with_false_split"]+r["runs_with_false_merge"] for r in cells]
        MACROS[f"DecBandWrongMin{word}"] = str(min(wrong))
        MACROS[f"DecBandWrongMax{word}"] = str(max(wrong))
        MACROS[f"DecBandFalseMerge{word}"] = str(sum(r["runs_with_false_merge"] for r in cells))
        whole = next(r for r in v6 if r.get("states") == states and r.get("method") == "bootstrap_band"
                     and "whole_run_successes" in r)
        MACROS[f"DecBandWhole{word}"] = str(whole["whole_run_successes"])
        cov = [r["coverage"] for r in cells]
        MACROS[f"DecBandCovMin{word}"] = f"{min(cov):.2f}"
        MACROS[f"DecBandCovMax{word}"] = f"{max(cov):.2f}"
    for states, word in [(16, "Small"), (128, "Largest")]:
        sup = next(r for r in v6 if r.get("states") == states and r.get("method") == "support"
                   and r.get("n_per_row") == 8192)
        base = next(r for r in v6 if r.get("states") == states and r.get("method") == "baseline"
                    and r.get("n_per_row") == 8192)
        MACROS[f"DecSupportMerge{word}"] = f"{100*sup['zero_pairs_merged']:.0f}"
        MACROS[f"DecBaseMerge{word}"] = f"{100*base['zero_pairs_merged']:.0f}"
    cert = [r for r in v6 if r.get("method") in ["baseline", "global", "support"] and "n_per_row" in r]
    MACROS["DecCertFalseRuns"] = str(sum(r["runs_with_false_merge"]+r["runs_with_false_split"] for r in cert))


def riverswim():
    rs, nav = load("riverswim"), load("navigation")
    if rs:
        MACROS["RsDiameter"] = f"{rs['true_diameter']:.3f}"
        lines = []
        cells = [c for c in rs["cells"] if "n_per_row" in c]
        for n in sorted({c["n_per_row"] for c in cells}):
            row = [f"{n:,}"]
            for method in ["baseline", "global", "support", "bootstrap_band"]:
                c = next(x for x in cells if x["method"] == method and x["n_per_row"] == n)
                row += [fmt(c["coverage"]), fmt(c["mean_sup_width"])]
            lines.append(" & ".join(row)+BS*2)
        write("riverswim_rows.tex", lines)
        whole = {c["method"]: c for c in rs["cells"] if "whole_run_successes" in c}
        for method, tag in [("baseline", "Base"), ("global", "Global"), ("support", "Support"), ("bootstrap_band", "Band")]:
            MACROS[f"Rs{tag}Whole"] = str(whole[method]["whole_run_successes"])
        final = {c["method"]: c for c in cells if c["n_per_row"] == 131072}
        MACROS["RsSupportRatio"] = f"{final['baseline']['mean_sup_width']/final['support']['mean_sup_width']:.1f}"
        MACROS["RsBandRatio"] = f"{final['support']['mean_sup_width']/final['bootstrap_band']['mean_sup_width']:.0f}"
        MACROS["RsSplitSupport"] = f"{100*final['support']['certified_split_fraction']:.0f}"
        MACROS["RsSplitBase"] = f"{100*final['baseline']['certified_split_fraction']:.0f}"
    if nav:
        by = {r["navigation"]: r for r in nav}
        MACROS["RsNavOptSteps"] = f"{by['optimistic']['mean']:,.0f}"
        MACROS["RsNavOptDone"] = str(by["optimistic"]["completed"])
        MACROS["RsNavRandDone"] = str(by["random"]["completed"])
        MACROS["RsNavRuns"] = str(by["random"]["runs"])
        MACROS["RsNavRandRetained"] = f"{by['random']['retained_samples']['mean']:,.0f}"
        MACROS["RsNavRandSteps"] = f"{by['random']['mean']:,.0f}"
        MACROS["RsNavRatio"] = f"{by['random']['mean']/by['optimistic']['mean']:.0f}"


def priorities():
    pr, tg = load("priorities"), load("targeted")
    human = {"corridor": "Corridor", "branching_return": "Branching return", "riverswim": "RiverSwim"}
    lines = []
    if pr:
        for g in pr["groups"]:
            keys = sorted([k for k in g if k.startswith("mean_width_at_")], key=lambda k: int(k.split("_")[-1]))
            row = [human[g["task"]], g["rule"].capitalize(), f"{g['steps']['mean']:,.0f}",
                   fmt(g[keys[-3]]["mean"]), fmt(g[keys[-2]]["mean"]), fmt(g[keys[-1]]["mean"]),
                   f"{g['completed']}/{g['runs']}"]
            lines.append(" & ".join(row)+BS*2)
        write("priority_rows.tex", lines)
        tags = {"corridor": "Corr", "branching_return": "Branch", "riverswim": "River"}
        kinds = {"adjoint minus source": "AdjSrc", "adjoint minus uniform": "AdjUni",
                 "source minus uniform": "SrcUni", "propagated minus source": "PropSrc"}
        budgets = {t: sorted(int(k.split("_")[-1]) for k in next(g for g in pr["groups"] if g["task"] == t)
                             if k.startswith("mean_width_at_")) for t in tags}
        for c in pr["contrasts"]:
            kind, tag = kinds[c["contrast"]], tags[c["task"]]
            if c["endpoint"] == "steps":
                name, digits = "Steps", 0
            else:
                b = int(c["endpoint"].split("_")[-1])
                if b == budgets[c["task"]][-1]:
                    name = "WidthLate"
                elif b == budgets[c["task"]][2]:
                    name = "WidthMid"
                else:
                    continue
                digits = 3
            for suffix, value in [("", c["mean"]), ("Lo", c["ci95"][0]), ("Hi", c["ci95"][1])]:
                MACROS[f"Pri{kind}{tag}{name}{suffix}"] = f"{value:.{digits}f}"
        for task, tag in tags.items():
            g = {r["rule"]: r for r in pr["groups"] if r["task"] == task}
            MACROS[f"PriSpread{tag}"] = f"{100*(max(r['steps']['mean'] for r in g.values())/min(r['steps']['mean'] for r in g.values())-1):.2f}"
    if tg:
        tlines = []
        for g in tg["groups"]:
            tlines.append(" & ".join([human[g["task"]], "Adjoint (target)" if g["rule"] == "adjoint_target" else "Count",
                                      f"{g['steps']['mean']:,.0f} [{g['steps']['ci95'][0]:,.0f}, {g['steps']['ci95'][1]:,.0f}]",
                                      f"{100*g['target_share']['mean']:.0f}", f"{g['certified']}/{g['runs']}"])+BS*2)
        write("targeted_rows.tex", tlines)
        for c in tg["contrasts"]:
            tag = {"corridor": "Corr", "riverswim": "River"}[c["task"]]
            MACROS[f"Tgt{tag}Diff"] = f"{c['mean']:.0f}"
            MACROS[f"Tgt{tag}DiffLo"] = f"{c['ci95'][0]:.0f}"
            MACROS[f"Tgt{tag}DiffHi"] = f"{c['ci95'][1]:.0f}"
        for task, tag in [("corridor", "Corr"), ("riverswim", "River")]:
            g = {r["rule"]: r for r in tg["groups"] if r["task"] == task}
            MACROS[f"Tgt{tag}Saving"] = f"{100*(1-g['adjoint_target']['steps']['mean']/g['count']['steps']['mean']):.0f}"


def neural():
    path = NEURAL/"summary.json"
    if not path.exists():
        return
    final = json.loads(path.read_text())["final"]
    lines = []
    for method, label in [("raw", "Raw observations (oracle scale)"), ("point", "Point targets"),
                          ("interval", "Interval targets (App.~F)")]:
        f = final[method]
        row = [label, fmt(f["mean_distortion"]["mean"]), fmt(f["sup_distortion"]["mean"]),
               fmt(f["same_latent_mean_distance"]["mean"]), fmt(f["different_latent_mean_distance"]["mean"])]
        row += [fmt(f["interval_coverage"]["mean"]), fmt(f["interval_mean_width"]["mean"])] if "interval_coverage" in f else ["--", "--"]
        lines.append(" & ".join(row)+BS*2)
    (NEURAL/"neural_rows.tex").write_text("\n".join(lines)+"\n"+BS+"bottomrule\n", encoding="utf8")
    for method, tag in [("raw", "Raw"), ("point", "Point"), ("interval", "Interval")]:
        f = final[method]
        MACROS[f"Nn{tag}Mean"] = fmt(f["mean_distortion"]["mean"])
        MACROS[f"Nn{tag}Same"] = fmt(f["same_latent_mean_distance"]["mean"])
        MACROS[f"Nn{tag}Diff"] = fmt(f["different_latent_mean_distance"]["mean"])
    for method, tag in [("raw", "Raw"), ("point", "Point"), ("interval", "Interval")]:
        f = final[method]
        MACROS[f"Nn{tag}Ratio"] = f"{f['different_latent_mean_distance']['mean']/f['same_latent_mean_distance']['mean']:.2f}"
    floor = json.loads((NEURAL/"bias_floor.json").read_text())["rows"]
    for row, word in zip(floor, ["Eight", "ThirtyTwo", "OneTwentyEight"]):
        MACROS[f"NnBias{word}"] = fmt(row["same_latent_target_mean"])
    f = final["interval"]
    MACROS["NnCoverage"] = fmt(f["interval_coverage"]["mean"])
    MACROS["NnCoverageMin"] = fmt(f["interval_coverage"]["min"])
    MACROS["NnWidth"] = fmt(f["interval_mean_width"]["mean"])


if __name__ == "__main__":
    failures = widths()
    decisions()
    riverswim()
    priorities()
    neural()
    for name in list(MACROS):
        if MACROS[name].startswith("-"):
            MACROS[name+"Abs"] = MACROS[name][1:]
    (DATA/"summary_macros.tex").write_text("\n".join(
        BS+"newcommand{"+BS+name+"}{"+value+"}" for name, value in MACROS.items())+"\n", encoding="utf8")
    print(json.dumps(MACROS, indent=2))
    print("coverage failures:", failures)
