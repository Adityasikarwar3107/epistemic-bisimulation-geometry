"""Tables and macros for V5-V7."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results"/"extended"
MACROS = {}


def load(name):
    return json.loads((DATA/f"summary_{name}.json").read_text())[name]


def write(name, lines):
    (DATA/name).write_text("\n".join(lines)+"\n"+chr(92)+"bottomrule\n", encoding="utf8")


def baselines():
    summary = load("baselines")
    cells = {(r["model"], r["n_per_row"], r["method"]): r for r in summary if "n_per_row" in r}
    whole = {(r["model"], r["method"]): r for r in summary if "whole_run_successes" in r}
    methods = ["certified", "global", "bootstrap_pointwise", "bootstrap_band"]
    lines = []
    for model, states, gamma in [("ring6_g0.6", 6, .6), ("ring8_g0.9", 8, .9)]:
        if lines:
            lines.append(chr(92)+"midrule")
        for n in [128, 2048, 8192, 131072]:
            row = [str(states), str(gamma), f"{n:,}"]
            for method in methods:
                r = cells[(model, n, method)]
                row += [f"{r['coverage']:.3f}", f"{r['mean_sup_width']:.3f}"]
            lines.append(" & ".join(row)+chr(92)*2)
    write("baseline_rows.tex", lines)
    for model, tag in [("ring6_g0.6", "Low"), ("ring8_g0.9", "High")]:
        for method, name in [("certified", "Cert"), ("bootstrap_band", "Band"),
                             ("bootstrap_pointwise", "Point")]:
            MACROS[f"Ext{name}Whole{tag}"] = str(whole[(model, method)]["whole_run_successes"])
        band = [cells[(model, n, "bootstrap_band")]["coverage"] for n in [128, 512, 2048, 8192, 32768, 131072]]
        point = [cells[(model, n, "bootstrap_pointwise")]["coverage"] for n in [128, 512, 2048, 8192, 32768, 131072]]
        MACROS[f"ExtBandMin{tag}"] = f"{min(band):.3f}"
        MACROS[f"ExtPointMin{tag}"] = f"{min(point):.3f}"
        MACROS[f"ExtPointMax{tag}"] = f"{max(point):.3f}"
        gains = [1-cells[(model, n, "certified")]["mean_sup_width"]/cells[(model, n, "global")]["mean_sup_width"]
                 for n in [128, 512, 2048, 8192, 32768, 131072]]
        MACROS[f"ExtCertGain{tag}"] = f"{100*max(gains):.0f}"
        final = {m: cells[(model, 131072, m)] for m in methods}
        MACROS[f"ExtBandWidthRatio{tag}"] = f"{final['certified']['mean_sup_width']/final['bootstrap_band']['mean_sup_width']:.1f}"
        MACROS[f"ExtGlobalWidthRatio{tag}"] = f"{final['global']['mean_sup_width']/final['certified']['mean_sup_width']:.2f}"
    return summary


def scaling():
    summary = load("scaling")
    cells = {(r["states"], r["n_per_row"]): r for r in summary if "n_per_row" in r}
    whole = {r["states"]: r for r in summary if "whole_run_successes" in r}
    states = sorted(whole)
    counts = sorted({n for _, n in cells})
    lines = []
    for s in states:
        final = cells[(s, counts[-1])]
        row = [str(s)] + [f"{cells[(s, n)]['mean_sup_width']:.3f}" for n in counts]
        row += [f"{final['mean_anchors']:.2f}", f"{final['max_anchor_distortion']:.3f}",
                f"{final['median_certificate_seconds']:.2f}"]
        lines.append(" & ".join(row)+chr(92)*2)
    write("scaling_rows.tex", lines)
    MACROS["ExtScalingRuns"] = str(sum(whole[s]["runs"] for s in states))
    MACROS["ExtScalingCovered"] = str(sum(whole[s]["whole_run_successes"] for s in states))
    MACROS["ExtScalingWidthSmall"] = f"{cells[(states[0], counts[-1])]['mean_sup_width']:.3f}"
    MACROS["ExtScalingWidthLarge"] = f"{cells[(states[-1], counts[-1])]['mean_sup_width']:.3f}"
    MACROS["ExtScalingSeparation"] = f"{cells[(states[0], counts[-1])]['true_block_separation']:.3f}"
    MACROS["ExtScalingTimeLarge"] = f"{cells[(states[-1], counts[-1])]['median_certificate_seconds']:.1f}"
    MACROS["ExtScalingErrorLarge"] = f"{cells[(states[-1], counts[-1])]['mean_error']:.3f}"
    return summary


def anytime():
    summary = load("anytime")
    groups = summary["groups"]
    posthoc = load("anytime_posthoc") if (DATA/"summary_anytime_posthoc.json").exists() else None
    human = {"corridor": "Corridor", "branching_return": "Branching return"}
    budgets = [20000, 40000, 80000]
    lines = []
    for task in ["corridor", "branching_return"]:
        for ties, source in [("First action", groups), ("Unresolved row", posthoc["groups"] if posthoc else [])]:
            for g in [r for r in source if r["task"] == task]:
                steps = g["steps_to_quota"]
                row = [human[task], g["priority"].capitalize(), ties,
                       f"{steps['mean']:,.0f} [{steps['ci95'][0]:,.0f}, {steps['ci95'][1]:,.0f}]"]
                row += [f"{g[f'mean_width_at_{b}']['mean']:.3f}" for b in budgets]
                row += [f"{g['completed']}/{g['runs']}"]
                lines.append(" & ".join(row)+chr(92)*2)
    write("anytime_rows.tex", lines)
    if posthoc:
        for task, tag in [("corridor", "Corr"), ("branching_return", "Branch")]:
            eager = {r["priority"]: r for r in posthoc["groups"] if r["task"] == task}
            pre = {r["priority"]: r for r in groups if r["task"] == task}
            MACROS[f"ExtEager{tag}UniformSteps"] = f"{eager['uniform']['steps_to_quota']['mean']:,.0f}"
            MACROS[f"ExtEager{tag}LocalSteps"] = f"{eager['local']['steps_to_quota']['mean']:,.0f}"
            MACROS[f"ExtEager{tag}PropSteps"] = f"{eager['propagated']['steps_to_quota']['mean']:,.0f}"
            MACROS[f"ExtEager{tag}Saving"] = f"{100*(1-eager['uniform']['steps_to_quota']['mean']/pre['uniform']['steps_to_quota']['mean']):.1f}"
            eby = {(c["contrast"], c["endpoint"]): c for c in posthoc["paired_contrasts"] if c["task"] == task}
            c = eby[("local minus uniform", "steps_to_quota")]
            MACROS[f"ExtEager{tag}LocUniSteps"] = f"{c['mean']:,.0f}"
            MACROS[f"ExtEager{tag}LocUniStepsLo"] = f"{c['ci95'][0]:,.0f}"
            MACROS[f"ExtEager{tag}LocUniStepsHi"] = f"{c['ci95'][1]:,.0f}"
            for budget, word in [(40000, "Mid"), (80000, "Late")]:
                c = eby[("local minus uniform", f"mean_width_at_{budget}")]
                MACROS[f"ExtEager{tag}LocUniWidth{word}"] = f"{c['mean']:.3f}"
                MACROS[f"ExtEager{tag}LocUniWidth{word}Lo"] = f"{c['ci95'][0]:.3f}"
                MACROS[f"ExtEager{tag}LocUniWidth{word}Hi"] = f"{c['ci95'][1]:.3f}"
                c = eby[("propagated minus local", f"mean_width_at_{budget}")]
                MACROS[f"ExtEager{tag}PropLocWidth{word}"] = f"{c['mean']:.3f}"
                MACROS[f"ExtEager{tag}PropLocWidth{word}Lo"] = f"{c['ci95'][0]:.3f}"
                MACROS[f"ExtEager{tag}PropLocWidth{word}Hi"] = f"{c['ci95'][1]:.3f}"
            spread = [eager[k]["steps_to_quota"]["mean"] for k in eager]
            MACROS[f"ExtEager{tag}Spread"] = f"{100*(max(spread)-min(spread))/min(spread):.2f}"
            c = eby[("propagated minus local", "steps_to_quota")]
            MACROS[f"ExtEager{tag}PropLocSteps"] = f"{c['mean']:,.0f}"
            MACROS[f"ExtEager{tag}PropLocStepsLo"] = f"{c['ci95'][0]:,.0f}"
            MACROS[f"ExtEager{tag}PropLocStepsHi"] = f"{c['ci95'][1]:,.0f}"
    by = {(c["task"], c["contrast"], c["endpoint"]): c for c in summary["paired_contrasts"]}
    for task, tag in [("corridor", "Corr"), ("branching_return", "Branch")]:
        g = {r["priority"]: r for r in groups if r["task"] == task}
        MACROS[f"ExtSteps{tag}Reduction"] = f"{100*(1-g['local']['steps_to_quota']['mean']/g['uniform']['steps_to_quota']['mean']):.1f}"
        for first, second, name in [("local", "uniform", "LocUni"), ("propagated", "local", "PropLoc")]:
            c = by[(task, f"{first} minus {second}", "steps_to_quota")]
            MACROS[f"Ext{tag}{name}Steps"] = f"{c['mean']:,.0f}"
            MACROS[f"Ext{tag}{name}StepsLo"] = f"{c['ci95'][0]:,.0f}"
            MACROS[f"Ext{tag}{name}StepsHi"] = f"{c['ci95'][1]:,.0f}"
            c = by[(task, f"{first} minus {second}", "mean_width_at_80000")]
            MACROS[f"Ext{tag}{name}Width"] = f"{c['mean']:.3f}"
            MACROS[f"Ext{tag}{name}WidthLo"] = f"{c['ci95'][0]:.3f}"
            MACROS[f"Ext{tag}{name}WidthHi"] = f"{c['ci95'][1]:.3f}"
    return summary


if __name__ == "__main__":
    for part in [baselines, scaling, anytime]:
        if (DATA/f"summary_{part.__name__}.json").exists():
            part()
    # Text macros quote magnitudes; signs are stated in words.
    for name in list(MACROS):
        if MACROS[name].startswith("-"):
            MACROS[name+"Abs"] = MACROS[name][1:]
    (DATA/"summary_macros.tex").write_text("\n".join(
        chr(92)+"newcommand{"+chr(92)+name+"}{"+value+"}" for name, value in MACROS.items())+"\n",
        encoding="utf8")
    print(json.dumps(MACROS, indent=2))
