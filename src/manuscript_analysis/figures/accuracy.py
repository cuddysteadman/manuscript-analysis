from __future__ import annotations

from functools import wraps
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from manuscript_analysis.figures.export import EXPORT_FORMATS, LABEL_VARIANTS
from manuscript_analysis.figures.style import apply_paper_style, paper_style_context


AUC_JITTER_SEED = 2048
PIPELINES = ("SORTED", "PRUNED", "UNSORTED")
DISPLAY_NAMES = ("Sorted", "Pruned", "Unsorted")
POINT_COLORS = {
    "SORTED": "#55a6d7",
    "PRUNED": "#9867d5",
    "UNSORTED": "#f05c43",
}
BOX_COLORS = {
    "SORTED": "#a9cfe4",
    "PRUNED": "#c9b5e5",
    "UNSORTED": "#f3b7a9",
}

ACCURACY_LAYOUT_STYLE = {
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 15,
    "ytick.labelsize": 14,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
}


def apply_style():
    apply_paper_style()
    mpl.rcParams.update(ACCURACY_LAYOUT_STYLE)


def _paper_styled(builder):
    @wraps(builder)
    def wrapped(*args, **kwargs):
        with paper_style_context():
            return builder(*args, **kwargs)

    return wrapped


def accuracy_by_seed(path):
    predictions = pd.read_csv(path)
    predictions["correct"] = (
        predictions["predicted_label"].astype(int)
        == predictions["class_label"].astype(int)
    )
    return (
        predictions.groupby(["seed", "pipeline"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )


def auc_by_seed(path, *, level):
    column = f"{level}_auc"
    metrics = pd.read_csv(path)
    required = {"pipeline", "seed", column}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            f"{path} is missing the required AUC columns: {sorted(missing)}"
        )
    return metrics.loc[:, ["seed", "pipeline", column]].rename(
        columns={column: "auc"}
    )


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_maximum = 0.0
    for rank, original_index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[original_index])
        running_maximum = max(running_maximum, value)
        adjusted[original_index] = running_maximum
    return adjusted


def significance_label(p_value, *, n_comparisons=1):
    if p_value < 0.001 / n_comparisons:
        return "***"
    if p_value < 0.01 / n_comparisons:
        return "**"
    if p_value < 0.05 / n_comparisons:
        return "*"
    return "n.s."


def paired_significance(data, *, value_column="accuracy", correction="holm"):
    comparisons = [
        ("SORTED", "PRUNED"),
        ("PRUNED", "UNSORTED"),
        ("SORTED", "UNSORTED"),
    ]
    wide = data.pivot(index="seed", columns="pipeline", values=value_column)
    raw = []
    for first, second in comparisons:
        paired = wide[[first, second]].dropna()
        raw.append(wilcoxon(
            paired[first],
            paired[second],
            alternative="two-sided",
            zero_method="wilcox",
        ).pvalue)
    if correction == "bonferroni":
        adjusted = [min(1.0, len(comparisons) * value) for value in raw]
        label_comparisons = 1
    elif correction == "holm":
        adjusted = holm_adjust(raw)
        label_comparisons = 1
    else:
        raise ValueError(
            "correction must be either 'bonferroni' or 'holm'"
        )
    return [
        {
            "first": first,
            "second": second,
            "raw_p": raw_p,
            "adjusted_p": adjusted_p,
            "label": significance_label(
                adjusted_p,
                n_comparisons=label_comparisons,
            ),
        }
        for (first, second), raw_p, adjusted_p
        in zip(comparisons, raw, adjusted)
    ]


def report_auc_significance(data):
    for result in paired_significance(data, value_column="auc", correction="bonferroni"):
        print(
            f"{result['first'].title()} vs {result['second'].title()}: "
            f"raw p = {result['raw_p']:.6g}; "
            f"Bonferroni-adjusted p = {result['adjusted_p']:.6g} "
            f"({result['label']})"
        )


def jitter_offsets(data, jitter_seed):
    rng = np.random.default_rng(jitter_seed)
    offsets = pd.Series(np.nan, index=data.index)
    for pipeline in PIPELINES:
        mask = data["pipeline"].eq(pipeline)
        offsets[mask] = rng.uniform(-0.055, 0.055, mask.sum())
    return offsets


def draw_boxplots(
    ax,
    data,
    jitter_seed,
    *,
    value_column="accuracy",
    ylabel=None,
    show_chance_line=True,
):
    jitter = jitter_offsets(data, jitter_seed)
    values_by_pipeline = [
        data.loc[data.pipeline.eq(pipeline), value_column].to_numpy()
        for pipeline in PIPELINES
    ]
    boxplot = ax.boxplot(
        values_by_pipeline,
        positions=np.arange(3),
        widths=0.35,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        boxprops={"linewidth": 1.4, "edgecolor": "black"},
        medianprops={"color": "black", "linewidth": 2.0},
        whiskerprops={"color": "black", "linewidth": 1.4},
        capprops={"color": "black", "linewidth": 1.4},
    )
    for patch, pipeline in zip(boxplot["boxes"], PIPELINES):
        patch.set_facecolor(BOX_COLORS[pipeline])
        patch.set_alpha(0.65)
    for x_position, pipeline, values in zip(
        np.arange(3), PIPELINES, values_by_pipeline
    ):
        ax.scatter(
            x_position + jitter[data.pipeline.eq(pipeline)].to_numpy(),
            values,
            s=24,
            facecolor=POINT_COLORS[pipeline],
            edgecolor=mpl.colors.to_rgba(POINT_COLORS[pipeline], 0.95),
            linewidth=0.7,
            alpha=0.85,
            zorder=3,
        )
    if show_chance_line:
        ax.axhline(0.5, color="black", linestyle="--", alpha=1.0)
    ax.set_xlim(-0.55, 2.85)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks(np.arange(3), labels=DISPLAY_NAMES)
    ax.set_yticks(np.arange(0.3, 1.01, 0.1))
    ax.set_ylabel(ylabel or "Test data accuracy", labelpad=6)
    ax.tick_params(axis="x", which="both", colors="black", length=0, pad=10)
    ax.tick_params(axis="y", which="both", colors="black", length=0, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def add_significance_bars(ax, data, *, value_column="accuracy", y_positions=None):
    results = paired_significance(data, value_column=value_column)
    x_positions = {"SORTED": 0, "PRUNED": 1, "UNSORTED": 2}
    y_positions = y_positions or {
        ("SORTED", "PRUNED"): 0.990,
        ("PRUNED", "UNSORTED"): 1.010,
        ("SORTED", "UNSORTED"): 1.052,
    }
    for result in results:
        first, second = result["first"], result["second"]
        x_start, x_end = x_positions[first], x_positions[second]
        y_position = y_positions[(first, second)]
        inset = 0.02 if x_end - x_start == 1 else 0.0
        ax.plot(
            [x_start + inset, x_end - inset],
            [y_position, y_position],
            color="black",
            linewidth=3.0,
            solid_capstyle="butt",
            clip_on=False,
            zorder=5,
        )
        fontsize = 16 if "*" in result["label"] else 11
        ax.text(
            (x_start + x_end) / 2,
            y_position + 0.006,
            result["label"],
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="normal",
            color="black",
            clip_on=False,
            zorder=6,
        )
    return results


def add_auc_significance_brackets(ax, data):
    results = paired_significance(
        data,
        value_column="auc",
        correction="bonferroni",
    )
    x_positions = {"SORTED": 0, "PRUNED": 1, "UNSORTED": 2}
    bar_base = max(1.035, float(data["auc"].max()) + 0.015)
    y_positions = {
        ("SORTED", "PRUNED"): bar_base,
        ("PRUNED", "UNSORTED"): bar_base + 0.020,
        ("SORTED", "UNSORTED"): bar_base + 0.060,
    }

    for result in results:
        first, second = result["first"], result["second"]
        x_start, x_end = x_positions[first], x_positions[second]
        y_position = y_positions[(first, second)]
        inset = 0.02 if x_end - x_start == 1 else 0.0
        ax.plot(
            [x_start + inset, x_end - inset],
            [y_position, y_position],
            color="black",
            linewidth=3.0,
            solid_capstyle="butt",
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (x_start + x_end) / 2,
            y_position + 0.006,
            result["label"],
            ha="center",
            va="bottom",
            fontsize=16 if "*" in result["label"] else 11,
            fontweight="normal",
            color="black",
            clip_on=False,
            zorder=6,
        )

    return results


@_paper_styled
def make_accuracy_figure(bundle_dir, output_path=None):
    apply_style()
    bundle_dir = Path(bundle_dir)
    output_path = Path(
        output_path or bundle_dir / "test_accuracy_boxplots.png"
    )
    block_accuracy = accuracy_by_seed(bundle_dir / "block_predictions_test.csv")
    session_accuracy = accuracy_by_seed(
        bundle_dir / "session_predictions_test.csv"
    )
    width, height = 1464, 718
    figure = plt.figure(
        figsize=(width / 100, height / 100),
        dpi=100,
        facecolor="white",
        edgecolor="white",
        layout=None,
    )
    left = figure.add_axes([
        132 / width, 44 / height, 598 / width, 566 / height,
    ])
    right = figure.add_axes([
        793 / width, 44 / height, 598 / width, 566 / height,
    ])
    draw_boxplots(left, block_accuracy, jitter_seed=1024)
    block_statistics = add_significance_bars(left, block_accuracy)
    draw_boxplots(right, session_accuracy, jitter_seed=2048)
    session_statistics = add_significance_bars(right, session_accuracy)
    figure.text(
        65 / width, 1 - 51 / height, "A", ha="left", va="top",
        fontsize=25, fontweight="bold", color="black",
    )
    figure.text(
        743 / width, 1 - 51 / height, "B", ha="left", va="top",
        fontsize=25, fontweight="bold", color="black",
    )
    figure.savefig(
        output_path,
        dpi=100,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    return output_path


@_paper_styled
def make_auc_figure(
    bundle_dir,
    *,
    level,
    output_path=None,
    show_chance_line=True,
    include_labels=True,
):
    apply_style()
    if level not in {"block", "session"}:
        raise ValueError("level must be either 'block' or 'session'")
    bundle_dir = Path(bundle_dir)
    output_path = Path(
        output_path or bundle_dir / f"test_{level}_auc_boxplot.png"
    )
    auc = auc_by_seed(bundle_dir / "run_metrics.csv", level=level)
    width, height = 642, 658
    figure = plt.figure(
        figsize=(width / 100, height / 100),
        facecolor="white",
        edgecolor="white",
        layout=None,
    )
    axis = figure.add_axes([
        112 / width, 54 / height, 506 / width, 526 / height,
    ])
    draw_boxplots(
        axis,
        auc,
        jitter_seed=AUC_JITTER_SEED,
        value_column="auc",
        ylabel="Test data AUC",
        show_chance_line=show_chance_line,
    )
    axis.set_ylim(0.20, 1.02)
    axis.set_yticks(np.arange(0.2, 1.01, 0.1))
    axis.spines["left"].set_color("black")
    axis.spines["bottom"].set_color("black")
    add_auc_significance_brackets(axis, auc)
    if not include_labels:
        for text_artist in figure.findobj(match=Text):
            text_artist.set_visible(False)
    figure.savefig(
        output_path,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    return output_path


def make_block_auc_figure(
    bundle_dir,
    output_path=None,
    *,
    show_chance_line=True,
    include_labels=True,
):
    return make_auc_figure(
        bundle_dir,
        level="block",
        output_path=output_path,
        show_chance_line=show_chance_line,
        include_labels=include_labels,
    )


def make_session_auc_figure(
    bundle_dir,
    output_path=None,
    *,
    show_chance_line=True,
    include_labels=True,
):
    return make_auc_figure(
        bundle_dir,
        level="session",
        output_path=output_path,
        show_chance_line=show_chance_line,
        include_labels=include_labels,
    )


def export_auc_boxplot(bundle_dir, output_dir, stem, *, level):
    bundle_dir, output_dir = Path(bundle_dir), Path(output_dir)
    data = auc_by_seed(bundle_dir / "run_metrics.csv", level=level)
    if set(data["pipeline"]) != set(PIPELINES):
        raise ValueError(
            f"Expected pipelines {sorted(PIPELINES)}, found {sorted(set(data['pipeline']))}"
        )
    data["display_name"] = data["pipeline"].map(dict(zip(PIPELINES, DISPLAY_NAMES)))
    data["pipeline_order"] = data["pipeline"].map({p: i for i, p in enumerate(PIPELINES)})
    data["jitter_offset"] = jitter_offsets(data, AUC_JITTER_SEED)
    data["plot_x"] = data["pipeline_order"] + data["jitter_offset"]
    data = data[["seed", "pipeline", "display_name", "pipeline_order", "auc", "jitter_offset", "plot_x"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    data.to_csv(csv_path, index=False)
    paths = [
        make_auc_figure(
            bundle_dir,
            level=level,
            output_path=output_dir / f"{stem}_{suffix}.{file_format}",
            include_labels=include_labels,
        )
        for suffix, include_labels in LABEL_VARIANTS
        for file_format in EXPORT_FORMATS
    ]
    return data, [*paths, csv_path]


@_paper_styled
def make_block_session_auc_figure(
    bundle_dir,
    output_path=None,
    *,
    show_chance_line=True,
):
    apply_style()
    bundle_dir = Path(bundle_dir)
    output_path = Path(output_path or bundle_dir / "test_auc_boxplots.png")
    block_auc = auc_by_seed(bundle_dir / "run_metrics.csv", level="block")
    session_auc = auc_by_seed(bundle_dir / "run_metrics.csv", level="session")
    width, height = 1464, 718
    figure = plt.figure(
        figsize=(width / 100, height / 100),
        facecolor="white",
        edgecolor="white",
        layout=None,
    )
    left, right = figure.subplots(1, 2)
    figure.subplots_adjust(
        left=132 / width,
        right=1391 / width,
        bottom=44 / height,
        top=610 / height,
        wspace=0.52,
    )
    for axis, auc, jitter_seed in (
        (left, block_auc, 1024),
        (right, session_auc, 2048),
    ):
        draw_boxplots(
            axis,
            auc,
            jitter_seed=jitter_seed,
            value_column="auc",
            ylabel="Test data AUC",
            show_chance_line=show_chance_line,
        )
        axis.set_ylim(0.20, 1.02)
        axis.set_yticks(np.arange(0.2, 1.01, 0.1))
        axis.spines["left"].set_color("black")
        axis.spines["bottom"].set_color("black")
        add_auc_significance_brackets(axis, auc)
    figure.text(
        65 / width, 1 - 51 / height, "A", ha="left", va="top",
        fontsize=26, fontweight="bold", color="black",
    )
    figure.text(
        841 / width, 1 - 51 / height, "B", ha="left", va="top",
        fontsize=26, fontweight="bold", color="black",
    )
    figure.savefig(
        output_path,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(figure)
    return output_path
