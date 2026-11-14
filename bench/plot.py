"""Turn bench/results/*.jsonl into the plots in docs/plots/.

    python bench/plot.py

Every number comes from the boss's own measurements (``mr status --json``): job time is
submission to completion, and the Gantt charts draw each task attempt from its recorded
start/end on the worker that ran it.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
PLOTS = REPO / "docs" / "plots"

# Palette (validated for color-vision deficiency): two categorical slots plus neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MAP_COLOR, REDUCE_COLOR = BLUE, ORANGE
WORKERS = ["worker1", "worker2", "worker3", "worker4"]

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)


# ---------------------------------------------------------------------- data


def load(experiment: str) -> list[dict]:
    path = RESULTS / f"{experiment}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_config(runs: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in runs:
        groups[r["config"][key]].append(r)
    return groups


def median_run(runs: list[dict]) -> dict:
    """The trial whose job time is the median: the representative run for a Gantt chart."""
    ordered = sorted(runs, key=lambda r: r["elapsed"])
    return ordered[(len(ordered) - 1) // 2]


def summary(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOTS / name)


def caption(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


def bar_with_range(ax, y: float, values: list[float], color: str, label_fmt: str) -> None:
    """Horizontal bar at the median, a thin min-max whisker, and each trial as a dot."""
    med, lo, hi = summary(values)
    ax.barh(y, med, height=0.5, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.plot([lo, hi], [y, y], color=INK_2, linewidth=1.2, zorder=3)
    ax.scatter(values, [y] * len(values), s=14, color=INK_2, zorder=4, linewidths=0)
    ax.text(
        hi + ax.get_xlim()[1] * 0.015,
        y,
        label_fmt.format(med),
        va="center",
        color=INK,
        fontsize=9.5,
    )


# ---------------------------------------------------------------------- plots


def plot_scaling() -> None:
    runs = by_config(load("scaling"), "workers")
    if not runs:
        return
    ns = sorted(runs)
    times = {n: [r["elapsed"] for r in runs[n]] for n in ns}
    t1 = statistics.median(times[ns[0]]) * ns[0]
    meds = [statistics.median(times[n]) for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.plot(ns, [t1 / n for n in ns], color=MUTED, linewidth=1.5, zorder=2, label="ideal (T₁ / n)")
    ax1.plot(
        ns,
        meds,
        label="measured (median)",
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=7,
        zorder=3,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    for n in ns:
        med, lo, hi = summary(times[n])
        ax1.plot([n, n], [lo, hi], color=BLUE, linewidth=1.2, zorder=3)
        ax1.annotate(
            f"{med:.1f} s",
            (n, med),
            textcoords="offset points",
            xytext=(8, 6),
            color=INK,
            fontsize=9,
        )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Job time vs. workers")
    ax1.set_xlabel("workers (1 CPU each)")
    ax1.set_ylabel("job time (s)")
    ax1.set_ylim(0, max(max(t) for t in times.values()) * 1.12)

    speedups = [meds[0] * ns[0] / m for m in meds]
    ax2.plot(ns, ns, color=MUTED, linewidth=1.5, zorder=2, label="ideal (linear)")
    ax2.plot(
        ns,
        speedups,
        label="measured (median)",
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=7,
        zorder=3,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    for n, s in zip(ns, speedups, strict=True):
        ax2.annotate(
            f"{s:.2f}×", (n, s), textcoords="offset points", xytext=(8, -14), color=INK, fontsize=9
        )
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title("Speedup over 1 worker")
    ax2.set_xlabel("workers (1 CPU each)")
    ax2.set_ylabel("speedup")
    ax2.set_ylim(0, ns[-1] + 0.6)
    for ax in (ax1, ax2):
        ax.set_xticks(ns)
        ax.set_xlim(ns[0] - 0.3, ns[-1] + 0.5)

    trials = len(times[ns[0]])
    caption(
        fig,
        f"Word count, 128 MiB synthetic corpus, 16 map / 4 reduce tasks, combiner on. "
        f"Median of {trials} trials; vertical whiskers span min–max.",
    )
    fig.tight_layout()
    save(fig, "scaling.png")


def plot_combiner() -> None:
    runs = by_config(load("combiner"), "combiner")
    if not runs:
        return
    setups = [(True, "with combiner", BLUE), (False, "without combiner", ORANGE)]
    metrics = [
        ("Job time", "seconds", lambda r: r["elapsed"], "{:.1f} s"),
        (
            "Data shuffled to reducers",
            "MiB",
            lambda r: r["counters"]["intermediate_bytes"] / 2**20,
            "{:.1f} MiB",
        ),
        (
            "Records shuffled to reducers",
            "millions",
            lambda r: r["counters"]["intermediate_records"] / 1e6,
            "{:.2f} M",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))
    for ax, (title, unit, fn, fmt) in zip(axes, metrics, strict=True):
        with_c = statistics.median(fn(r) for r in runs[True])
        without = statistics.median(fn(r) for r in runs[False])
        title += f"  ({without / with_c:.0f}× less)" if without / with_c >= 2 else ""
        hi = max(fn(r) for rs in runs.values() for r in rs)
        ax.set_xlim(0, hi * 1.35)
        for y, (cfg, _, color) in enumerate(setups):
            bar_with_range(ax, y, [fn(r) for r in runs[cfg]], color, fmt)
        ax.set_yticks(range(len(setups)), [name for _, name, _ in setups])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    trials = len(runs[True])
    caption(
        fig,
        "Word count, 128 MiB, 16 maps / 4 reduces, 4 workers. Bars: median of "
        f"{trials} trials; dots: individual trials. Output is identical either way.",
    )
    fig.tight_layout()
    save(fig, "combiner.png")


def gantt(ax, run: dict, title: str, xmax: float, marker: tuple[float, str] | None = None):
    """One lane per worker; one bar per task attempt."""
    lanes = {w: i for i, w in enumerate(WORKERS)}
    for a in run["attempts"]:
        y = lanes.get(a["worker_id"])
        if y is None or not a["end"]:
            continue
        state = a["state"].removeprefix("ATTEMPT_")
        color = MAP_COLOR if a["kind"] == "MAP" else REDUCE_COLOR
        wasted = state in ("KILLED", "LOST", "FAILED")
        ax.barh(
            y,
            a["end"] - a["start"],
            left=a["start"],
            height=0.62,
            color=color,
            alpha=0.28 if wasted else 1.0,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=2,
        )
        width = a["end"] - a["start"]
        label = {"KILLED": "cancelled", "LOST": "lost"}.get(state, "")
        if a["speculative"]:
            label = "backup" + (" (lost race)" if state == "KILLED" else "")
        if label and width > xmax * 0.06:
            ax.text(
                a["start"] + width / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=INK if wasted else "#ffffff",
                zorder=3,
            )
    ax.axvline(run["elapsed"], color=INK_2, linewidth=1, zorder=1)
    ax.text(
        run["elapsed"], -0.62, f" done {run['elapsed']:.1f} s", color=INK, fontsize=9, va="bottom"
    )
    if marker:
        x, text = marker
        ax.axvline(x, color="#d03b3b", linewidth=1.2, zorder=4)
        ax.text(x, -0.53, f" ✕ {text}", color="#d03b3b", fontsize=9, va="center", fontweight="bold")
    ax.set_yticks(range(len(WORKERS)), WORKERS)
    ax.set_ylim(len(WORKERS) - 0.4, -0.75)
    ax.set_xlim(0, xmax)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def gantt_legend(fig, extra: list[Patch] | None = None) -> None:
    handles = [
        Patch(color=MAP_COLOR, label="map attempt"),
        Patch(color=REDUCE_COLOR, label="reduce attempt"),
        Patch(color=MAP_COLOR, alpha=0.28, label="wasted attempt (cancelled / lost)"),
        *(extra or []),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        ncol=len(handles),
        fontsize=9,
        bbox_to_anchor=(1.0, 1.02),
    )


def plot_speculation() -> None:
    runs = by_config(load("speculation"), "setup")
    if not runs:
        return
    order = ["no straggler", "straggler, no speculation", "straggler + speculation"]
    order = [o for o in order if o in runs]

    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.set_xlim(0, max(r["elapsed"] for rs in runs.values() for r in rs) * 1.25)
    for y, setup in enumerate(order):
        color = BLUE if setup != "straggler, no speculation" else ORANGE
        bar_with_range(ax, y, [r["elapsed"] for r in runs[setup]], color, "{:.1f} s")
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("job time (s)")
    ax.set_title("Speculative execution vs. a straggler worker")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    trials = len(runs[order[0]])
    caption(
        fig,
        "Word count, 128 MiB, 8 maps / 4 reduces, 4 workers; the straggler is worker4 "
        "throttled to 0.25 CPU with `docker update --cpus 0.25`. Bars: median of "
        f"{trials} trials; dots: individual trials.",
    )
    fig.tight_layout()
    save(fig, "speculation.png")

    panels = [s for s in order if s.startswith("straggler")]
    reps = {s: median_run(runs[s]) for s in panels}
    xmax = max(r["elapsed"] for r in reps.values()) * 1.08
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.3 * len(panels)), sharex=True)
    for ax, setup in zip(axes, panels, strict=True):
        gantt(ax, reps[setup], f"{setup} (worker4 at 0.25 CPU)", xmax)
    axes[-1].set_xlabel("seconds since job submission")
    gantt_legend(fig)
    caption(
        fig,
        "Median trial of each setup. Without speculation, the job waits for "
        "worker4's slow tasks. With it, idle workers run backup copies, the first copy "
        "to finish is committed, and the slower copy is cancelled.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "speculation_gantt.png")


def plot_failure() -> None:
    runs = by_config(load("failure"), "setup")
    if not runs or "worker2 killed" not in runs:
        return
    base, killed = median_run(runs["no failure"]), median_run(runs["worker2 killed"])
    xmax = max(base["elapsed"], killed["elapsed"]) * 1.1
    tb = summary([r["elapsed"] for r in runs["no failure"]])
    tk = summary([r["elapsed"] for r in runs["worker2 killed"]])

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    gantt(axes[0], base, f"No failure (median job time {tb[0]:.1f} s)", xmax)
    gantt(
        axes[1],
        killed,
        f"worker2 killed (median job time {tk[0]:.1f} s)",
        xmax,
        marker=(killed["killed_at"], "worker2 killed"),
    )
    axes[-1].set_xlabel("seconds since job submission")
    gantt_legend(fig)
    trials = len(runs["no failure"])
    caption(
        fig,
        "Word count, 128 MiB, 16 maps / 4 reduces. `docker compose kill worker2` "
        f"{killed['killed_at']:.0f} s after submission. The boss declares it dead after "
        "5 s without heartbeats and reruns its in-flight task elsewhere; its finished "
        "map output is on the shared volume and is reused. Median trial of "
        f"{trials}; every run produced identical output.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "failure_gantt.png")


def main() -> None:
    plot_scaling()
    plot_combiner()
    plot_speculation()
    plot_failure()


if __name__ == "__main__":
    main()
