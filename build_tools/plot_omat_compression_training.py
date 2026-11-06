"""Plot measured OMat compression training results without rerunning models."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/omat24/compression-demo"
ORDER = ("full_pool", "envtok_T2", "envtok_T100", "random")
LABELS = ("Full\npool", "EnvTok\nT = 2", "EnvTok\nT = 100", "Random\nsame ceiling")
COLORS = ("#4b5563", "#0072b2", "#009e73", "#d55e00")


def plot(source: Path, output: Path) -> None:  # noqa: PLR0912, PLR0915 - four explicit scientific panels
    """Render retention, timing, held-out error and seed-zero fitting curves."""
    data: dict[str, Any] = json.loads(source.read_text())
    experiment = data["experiment"]
    selections = {row["selector"]: row for row in experiment["selections"]}
    runs = data["runs"]
    if not runs:
        raise ValueError("at least one completed training run is needed")
    grouped = {name: [row for row in runs if row["selector"] == name] for name in ORDER}
    seeds = sorted({row["seed"] for row in runs})
    steps = runs[0]["worker"]["training"]["steps"]
    complete = "summary" in data and all(
        {row["seed"] for row in rows} == set(seeds) for rows in grouped.values()
    )
    if not all(row["worker"]["training"]["steps"] == steps for row in runs):
        raise ValueError("plotted models must have matching epoch counts")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#a6abb1",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#1f2937",
            "axes.labelcolor": "#374151",
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(
        left=0.075, right=0.975, top=0.84, bottom=0.18, hspace=0.47, wspace=0.28
    )
    title = "OMat24: measured compression and MACE training"
    if not complete:
        title += " (partial results)"
    fig.suptitle(title, x=0.075, y=0.97, ha="left", fontsize=19, weight="bold")
    fig.text(
        0.075,
        0.92,
        f"{experiment['training_configurations']} training configurations; "
        f"{experiment['holdout_configurations']} parent-disjoint "
        "held-out configurations. "
        f"{steps:,} full-batch epochs; {len(seeds)} model seed(s).",
        fontsize=11,
    )
    fig.text(
        0.075,
        0.885,
        f"Native small periodic cells; {len(experiment['species'])} elements. "
        f"Fresh {runs[0]['worker']['parameter_count']:,}-parameter CPU models; "
        "selections fitted on training data only.",
        fontsize=10,
        color="#586270",
    )
    x = np.arange(len(ORDER))
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(x, LABELS)
        ax.set_xlim(-0.5, 3.5)
        ax.tick_params(axis="x", length=0, pad=8)

    ax = axes[0, 0]
    atoms = [selections[name]["training_atoms"] for name in ORDER]
    ax.bar(x, atoms, color=COLORS, width=0.6)
    ax.set_title("a  Training atoms retained", loc="left", pad=12)
    ax.set_ylabel("Atoms per epoch")
    ax.set_ylim(0, max(atoms) * 1.28)
    for pos, value in zip(x, atoms, strict=True):
        removed = 100 * (1 - value / atoms[0])
        note = "full reference" if pos == 0 else f"{removed:.1f}% fewer"
        ax.text(
            pos,
            value + max(atoms) * 0.035,
            f"{value:,}\n{note}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[0, 1]
    ax.set_title("b  Measured optimizer time", loc="left", pad=12)
    ax.set_ylabel("Seconds · median across model seeds")
    maxima = []
    for pos, name, color in zip(x, ORDER, COLORS, strict=True):
        values = [row["worker"]["fit_seconds"] for row in grouped[name]]
        if not values:
            continue
        median = float(np.median(values))
        maxima.append(max(values))
        ax.bar(pos, median, color=color, width=0.6)
        ax.scatter(
            pos + (np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else 0),
            values,
            color="#111827",
            s=14,
            zorder=3,
        )
        ax.annotate(
            f"{median:.1f} s",
            (pos, max(values)),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, max(maxima) * 1.25)

    ax = axes[1, 0]
    ax.set_title("c  Shared held-out force error", loc="left", pad=12)
    ax.set_ylabel("Force component RMSE (eV/Å)")
    zero = experiment["holdout_zero_force_rmse_eV_per_Ang"]
    ax.axhline(
        zero,
        color="#8c5660",
        linestyle="--",
        linewidth=1.2,
        label="Zero-force prediction",
    )
    full_values = [
        row["holdout"]["aggregate"]["forces"]["rmse"] for row in grouped["full_pool"]
    ]
    if full_values:
        ax.axhline(
            np.mean(full_values),
            color=COLORS[0],
            linestyle=":",
            linewidth=1.2,
            label="Full-pool mean",
        )
    maxima = [zero]
    for pos, name, color in zip(x, ORDER, COLORS, strict=True):
        values = [
            row["holdout"]["aggregate"]["forces"]["rmse"] for row in grouped[name]
        ]
        if not values:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        maxima.append(max(*values, mean + sd))
        ax.errorbar(
            pos,
            mean,
            yerr=sd,
            color=color,
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=2,
            zorder=4,
        )
        ax.scatter(
            pos + (np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else 0),
            values,
            color=color,
            s=15,
            alpha=0.6,
            zorder=3,
        )
        ax.annotate(
            f"{mean:.3f}",
            (pos, mean + sd),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, max(maxima) * 1.32)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axes[1, 1]
    ax.set_title("d  Training force error · model seed 0", loc="left", pad=12)
    ax.set_xlabel("Full-batch epoch")
    ax.set_ylabel("Training force component RMSE (eV/Å)")
    for name, label, color in zip(ORDER, LABELS, COLORS, strict=True):
        seed_zero = next((row for row in grouped[name] if row["seed"] == 0), None)
        if seed_zero is None:
            continue
        worker = seed_zero["worker"]
        trace = np.sqrt(np.asarray(worker["force_loss"]))
        final = worker["training_metrics"]["final"]["forces"]["rmse"]
        # Loss step 1 is measured before update 1; append the post-final metric.
        values = np.r_[trace, final]
        ax.plot(
            np.arange(len(values)),
            values,
            color=color,
            linewidth=1.5,
            label=label.replace("\n", " "),
        )
    ax.set_xlim(0, steps)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(
        0.075,
        0.055,
        "Error bars: mean ± sample SD across model initializations, "
        "not confidence intervals. "
        "Timing excludes selection and setup.\n"
        "Training curves use different training subsets. "
        "Fixed epochs do not establish convergence; "
        "this small development split does not establish simulation accuracy.",
        fontsize=9,
        color="#586270",
        linespacing=1.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=100,
        facecolor="white",
        metadata={
            "Description": "Input SHA-256: "
            + hashlib.sha256(source.read_bytes()).hexdigest()
        },
    )
    plt.close(fig)
    print(output)


def main() -> None:
    """Parse result and chart paths, then render the measured comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXAMPLE / "training-results.json")
    parser.add_argument("--output", type=Path, default=EXAMPLE / "training.png")
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
