"""Figures, tables, and manuscript macros for V1-V4 from the stored records."""
from pathlib import Path
import csv
import itertools
import json
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter
import pymupdf

from ebg import greedy_cover, embedding_distance
from run_verified_study import bootstrap_mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results"/"verified"
SUMMARY = json.loads((DATA/"summary.json").read_text())
BLUE, ORANGE, GREEN, GRAY = "#287DB5", "#DD7C20", "#199D68", "#6D7479"
COLORS = [BLUE, ORANGE, GREEN]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.1,
    "axes.titlesize": 9.1, "axes.labelsize": 8.3,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "axes.linewidth": .7, "legend.fontsize": 7.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white"})


def read(stem):
    return list(csv.DictReader((DATA/f"{stem}.csv").open(newline="", encoding="utf8")))


def axes_pair():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.85, wspace=.35)
    for ax in axes:
        ax.grid(axis="both", color="#D4D8DC", lw=.55, ls="--", alpha=.75)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)
    return fig, axes


FIGURES = []


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            # Matplotlib constructs extra tick labels outside explicit axis limits.
            if text in [label for ax in fig.axes for label in
                        ax.get_xticklabels()+ax.get_yticklabels()+
                        ax.get_xticklabels(minor=True)+ax.get_yticklabels(minor=True)]:
                continue
            outside.append(text.get_text())
    assert not outside, (stem, outside)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT/"figures"/f"{stem}.{ext}", dpi=480)
    with pymupdf.open(ROOT/"figures"/f"{stem}.pdf") as pdf:
        assert not pdf[0].get_images(), "Scientific figure unexpectedly rasterized"
    FIGURES.append({"stem": stem, "outside_canvas_labels": outside, "pdf_raster_images": 0})
    plt.close(fig)


def calibration():
    records = read("calibration")
    fig, axes = axes_pair()
    table = []
    for ax, (states, gamma) in zip(axes, [(6, .6), (8, .9)]):
        rows = [r for r in records if int(r["states"]) == states]
        counts = sorted({int(r["n_per_row"]) for r in rows if int(r["n_per_row"]) > 0})
        for key, label, color in [("sup_width", "Interval width", BLUE),
                                  ("midpoint_sup_error", "Midpoint error", ORANGE)]:
            stats = [bootstrap_mean([float(r[key]) for r in rows if int(r["n_per_row"]) == n])
                     for n in counts]
            means = np.array([s["mean"] for s in stats])
            ax.plot(counts, means, "o-", color=color, ms=3, lw=1.15, label=label)
            ax.fill_between(counts, [s["ci95"][0] for s in stats],
                            [s["ci95"][1] for s in stats], color=color, alpha=.18, lw=0)
            if key == "sup_width":
                ax.plot(counts, means/2, "--", color=GREEN, lw=1, label="Half-width")
        ax.set(xscale="log", ylim=(0, 1.04), xlabel="Generative samples per row",
               ylabel="Supremum diagnostic")
        ax.set_title(f"({'a' if states == 6 else 'b'}) $S={states}$, $\\gamma={gamma}$", loc="left")
        for n in [0, 2048, 8192, 131072]:
            group = [r for r in rows if int(r["n_per_row"]) == n]
            table.append(f"{states} & {gamma} & {n:,} & "
                         f"{np.mean([float(r['sup_width']) for r in group]):.3f} & "
                         f"{np.mean([float(r['midpoint_sup_error']) for r in group]):.3f} " + r"\\")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    (DATA/"calibration_rows.tex").write_text("\n".join(table)+"\n"+chr(92)+"bottomrule\n", encoding="utf8")
    save(fig, "verified_calibration")


def propagation():
    records = read("propagation")
    fig, axes = axes_pair()
    etas = [.02, .1, .3]
    end = [r for r in records if int(r["n_per_uncertain_row"]) == 8192]
    for key, label, color, style in [
            ("width_R", "Actual width", BLUE, "o-"),
            ("pair_majorant_R", "Pair majorant", ORANGE, "s--"),
            ("global_bound", "Global bound", GRAY, "^:")]:
        stats = [bootstrap_mean([float(r[key]) for r in end if float(r["eta"]) == eta]) for eta in etas]
        means = [s["mean"] for s in stats]
        axes[0].plot(etas, means, style, color=color, lw=1.15, ms=4,
                     markerfacecolor="white" if key == "pair_majorant_R" else color, label=label)
        axes[0].errorbar(etas, means, yerr=np.array([
            [s["mean"]-s["ci95"][0] for s in stats],
            [s["ci95"][1]-s["mean"] for s in stats]]), fmt="none", color=color,
            capsize=2, elinewidth=.7)
    axes[0].axhline(0, color="#AE3540", lw=.8, ls="--", label="Local source only")
    axes[0].set(xlabel="Pair escape probability", ylabel="Width / upper bound", ylim=(-.005, .17))
    axes[0].set_xticks(etas)
    axes[0].set_title("(a) 8192 samples per future row", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, 1.01), ncol=4, fontsize=7.1)
    for eta, color in zip(etas, COLORS):
        group = [r for r in records if float(r["eta"]) == eta
                 and int(r["n_per_uncertain_row"]) > 0]
        counts = sorted({int(r["n_per_uncertain_row"]) for r in group})
        bounds = [np.mean([float(r["pair_majorant_R"]) for r in group
                           if int(r["n_per_uncertain_row"]) == n]) for n in counts]
        stats = [bootstrap_mean([float(r["width_R"]) for r in group
                                 if int(r["n_per_uncertain_row"]) == n]) for n in counts]
        widths = [s["mean"] for s in stats]
        axes[1].plot(counts, bounds, color=color, lw=1.2, label=f"$\\eta={eta:.2f}$")
        axes[1].scatter(counts, widths, color=color, s=13, marker="o", zorder=4)
        axes[1].fill_between(counts, [s["ci95"][0] for s in stats],
                            [s["ci95"][1] for s in stats], color=color, alpha=.15, lw=0)
    axes[1].set(xscale="log", xlabel="Samples per uncertain future row", ylabel="Current-pair diagnostic")
    axes[1].set_title("(b) Line: majorant; marker: width", loc="left")
    axes[1].legend(frameon=False)
    save(fig, "verified_propagation")


def ablation():
    fig, axes = axes_pair()
    table = []
    for ax, task, title in zip(axes, ["corridor", "branching_return"], ["Corridor", "Branching return"]):
        for offset, mode, color in [(-.16, "local", BLUE), (.16, "propagated", ORANGE)]:
            groups = [next(r for r in SUMMARY["ablations"]["groups"]
                           if r["task"] == task and r["priority"] == mode and r["navigation"] == nav)
                      for nav in ["random", "optimistic"]]
            values = np.array([r["mean"] for r in groups])
            error = np.array([[r["mean"]-r["ci95"][0] for r in groups],
                              [r["ci95"][1]-r["mean"] for r in groups]])
            ax.bar(np.arange(2)+offset, values, width=.28, color=color, alpha=.9,
                   yerr=error, capsize=3, error_kw={"elinewidth": .8, "capthick": .7},
                   label="Source priorities" if mode == "local" else "Propagated priorities")
            for row in groups:
                human = "Corridor" if task == "corridor" else "Branching return"
                mean = f"{row['mean']:,.1f}"
                ci = f"[{row['ci95'][0]:,.1f}, {row['ci95'][1]:,.1f}]"
                table.append(f"{human} & {mode.capitalize()} & {row['navigation'].capitalize()} & "
                             f"{mean} {ci} & {row['completed']}/{row['runs']} " + r"\\")
        ax.set(yscale="log", ylabel="Interaction steps (log scale)", xlabel="Navigation policy")
        ax.set_xticks([0, 1], ["Random", "Optimistic"])
        ax.set_title(f"({'a' if task == 'corridor' else 'b'}) {title}", loc="left")
        ax.set_ylim(bottom=1000)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([1000, 3000, 10000, 30000] if task == "corridor" else [1000, 2000, 3000])
        ax.yaxis.set_major_formatter(ScalarFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False)
    (DATA/"ablation_rows.tex").write_text("\n".join(table)+"\n"+chr(92)+"bottomrule\n", encoding="utf8")
    save(fig, "verified_navigation_ablation")


def clocks():
    rows = read("clock_budgets")
    fig, axes = axes_pair()
    for dimension, color in zip([1, 2, 3], COLORS):
        group = [r for r in rows if int(r["d"]) == dimension]
        x = [1/float(r["epsilon"]) for r in group]
        for ax, key in zip(axes, ["cover_queries", "resolution_query_budget"]):
            ax.loglog(x, [int(r[key]) for r in group], "o-", color=color, ms=3.5, lw=1.1,
                      label=f"d = {dimension}")
    for ax, title, ylabel in zip(axes, ["(a) Cover visits", "(b) Prescribed confidence budget"],
                                 ["Queries: M + 1", "Queries: M n + 1"]):
        ax.set(xlabel="Inverse resolution", ylabel=ylabel)
        ax.set_xticks([128, 192, 256, 384])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    save(fig, "verified_clock_budgets")


def clock_table():
    """V4 budget rows of the manuscript table."""
    rows = read("clock_budgets")
    by = {(int(r["d"]), int(r["m"])): r for r in rows}
    dims = sorted({d for d, _ in by})
    sizes = sorted({m for _, m in by})

    def sci(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}{{\\times}}10^{{{int(exponent)}}}$"
    lines = []
    for m in sizes:
        inverse = 1/float(by[(dims[0], m)]["epsilon"])
        assert abs(inverse-32*m) < 1e-9
        cells = [str(m), str(32*m)]
        for d in dims:
            r = by[(d, m)]
            assert int(r["M"]) == m**d and int(r["cover_queries"]) == m**d+1
            cells += [str(int(r["cover_queries"])), str(int(r["n_per_unknown_row"])),
                      sci(int(r["resolution_query_budget"]))]
        lines.append(" & ".join(cells)+chr(92)*2)
    (DATA/"clock_rows.tex").write_text("\n".join(lines)+"\n"+chr(92)+"bottomrule\n", encoding="utf8")


def anchor_audit():
    archive = np.load(DATA/"calibration_0_n131072.npz")
    truth = archive["true_metric"]
    records = []
    for seed, upper, lower in zip(archive["seeds"], archive["upper"], archive["lower"]):
        width, radius = float(np.max(upper-lower)), .25
        assert width < radius
        centers = greedy_cover(upper, radius)
        pair_distance = embedding_distance(upper[:, centers])
        cover_radius = float(np.min(upper[:, centers], axis=1).max())
        distortion = float(np.max(np.abs(pair_distance-truth)))
        assert cover_radius <= radius+2e-9
        assert np.all(pair_distance <= upper+2e-9)
        assert np.all(pair_distance >= upper-2*radius-2e-9)
        assert distortion <= max(width, 2*radius)+2e-9
        assert np.max(np.abs(embedding_distance(upper)-upper)) < 2e-9
        threshold = (radius-width)/2
        true_cover = len(upper)
        for count in range(1, len(upper)+1):
            if any(np.min(truth[:, list(c)], axis=1).max() <= threshold+1e-12
                   for c in itertools.combinations(range(len(upper)), count)):
                true_cover = count
                break
        assert len(centers) <= true_cover
        records.append({"seed": int(seed), "radius": radius, "width": width,
                        "centers": len(centers), "cover_radius": cover_radius,
                        "infinity_distortion": distortion, "distortion_bound": max(width, 2*radius),
                        "true_cover_at_packing_scale": true_cover})
    with (DATA/"anchor_audit.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def numbers():
    anchors = anchor_audit()
    end = {row["model"]: row for row in SUMMARY["calibration"] if row.get("n_per_row") == 131072}
    coverage = next(row for row in SUMMARY["calibration"] if "coverage_ci95" in row)
    prop = [r for r in read("propagation") if float(r["eta"]) == .02
            and int(r["n_per_uncertain_row"]) == 8192]
    assert all(int(r["covered_R"]) == 1 for r in read("propagation"))
    values = {"VerifiedCoverageLower": f"{coverage['coverage_ci95'][0]:.4f}",
        "VerifiedLowWidth": f"{end['ring6_g0.6']['mean_width']:.3f}",
        "VerifiedLowError": f"{end['ring6_g0.6']['mean_error']:.4f}",
        "VerifiedHighWidth": f"{end['ring8_g0.9']['mean_width']:.3f}",
        "VerifiedHighError": f"{end['ring8_g0.9']['mean_error']:.4f}",
        "VerifiedAnchorCount": f"{np.mean([r['centers'] for r in anchors]):.2f}",
        "VerifiedAnchorMaxDistortion": f"{max(r['infinity_distortion'] for r in anchors):.3f}",
        "VerifiedPropagationRatio": f"{np.mean([float(r['global_bound']) for r in prop])/np.mean([float(r['pair_majorant_R']) for r in prop]):.2f}"}
    for task, name in [("corridor", "VerifiedCorridorNavigationRatio"),
                       ("branching_return", "VerifiedBranchNavigationRatio")]:
        groups = {r["navigation"]: r for r in SUMMARY["ablations"]["groups"]
                  if r["task"] == task and r["priority"] == "local"}
        values[name] = f"{groups['random']['mean']/groups['optimistic']['mean']:.2f}"
    (DATA/"summary_macros.tex").write_text("\n".join(
        chr(92)+"newcommand{"+chr(92)+name+"}{"+value+"}" for name, value in values.items())+"\n", encoding="utf8")
    contrasts = SUMMARY["ablations"]["paired_priority_contrasts"]
    text = "For random navigation, priority changes have no effect by construction. "
    for contrast in contrasts:
        if contrast["navigation"] != "optimistic":
            continue
        human = "corridor" if contrast["task"] == "corridor" else "branching-return task"
        text += (f"On the {human}, the paired change in mean steps (propagated minus source-only) is "
                 f"${contrast['mean']:.3f}$ with a 95\\% seed-bootstrap interval "
                 f"$[{contrast['ci95'][0]:.3f},{contrast['ci95'][1]:.3f}]$. ")
    text += "These tasks do not demonstrate a systematic acquisition advantage from propagated priorities.\n"
    (DATA/"priority_result.tex").write_text(text, encoding="utf8")
    return values


if __name__ == "__main__":
    macros = numbers()
    for function in [calibration, propagation, ablation, clocks, clock_table]:
        function()
    report = {"figures": FIGURES, "manuscript_numbers": macros,
              "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in DATA.glob("*.csv")}, "new_measurements_created_by_plotting": False}
    (ROOT/"figures"/"figure_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(report, indent=2))
