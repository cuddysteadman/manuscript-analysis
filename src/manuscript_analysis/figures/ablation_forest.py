from __future__ import annotations

from pathlib import Path
import statistics as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from manuscript_analysis.contract import N_RUNS, REFERENCE_PIPELINE

from manuscript_analysis.figures.fig_utils import (
    COLOR_ALL,
    COLOR_DROP,
    COLOR_ONLY,
    LABELS,
    MARKER_ALL,
    MARKER_DROP,
    MARKER_ONLY,
    MATCHED_THREE_EXPORTS,
    ablation_figure_set_look,
    get_out_path,
)
from manuscript_analysis.figures.style import paper_style_context


FAMILY_ALPHA = 0.05 / 4


def permutation_p_value(null_medians, observed):
    return (1 + np.count_nonzero(null_medians >= observed)) / (len(null_medians) + 1)


def _load_inputs(ablation_csv, null_csv, *, value_column, null_value_column):
    ablation_csv, null_csv = Path(ablation_csv), Path(null_csv)
    ablation = pd.read_csv(ablation_csv)
    null = pd.read_csv(null_csv)

    required_ablation = {"feature_set", value_column}
    missing_ablation = required_ablation - set(ablation)
    if missing_ablation:
        raise ValueError(
            f"{ablation_csv} is missing columns: "
            f"{', '.join(sorted(missing_ablation))}"
        )
    if null_value_column not in null:
        raise ValueError(
            f"{null_csv} is missing column: {null_value_column}"
        )

    unknown = set(ablation["feature_set"].dropna().astype(str)) - set(LABELS)
    if unknown:
        raise ValueError(
            "Unrecognized ablation feature sets: "
            f"{', '.join(sorted(unknown))}"
        )
    null_medians = pd.to_numeric(
        null[null_value_column], errors="raise"
    ).dropna().to_numpy(dtype=float)
    if not len(null_medians):
        raise ValueError(
            f"{null_csv} contains no values for {null_value_column}"
        )
    if not np.isfinite(null_medians).all():
        raise ValueError(f"{null_csv} contains non-finite null median AUCs")
    return ablation, null_medians


def _require_seed_coverage(frame, *, context):
    if "seed" not in frame:
        raise ValueError(f"{context} is missing column: seed")
    seeds = pd.to_numeric(frame["seed"], errors="raise")
    expected = set(range(N_RUNS))
    if len(frame) != N_RUNS or set(seeds) != expected:
        raise ValueError(
            f"{context} must contain exactly one row for every seed "
            f"0 through {N_RUNS - 1}"
        )
    if seeds.duplicated().any():
        raise ValueError(f"{context} contains duplicate seeds")
    return seeds.astype(int)


def _reuse_matched_three_baseline(
    ablation,
    *,
    value_column,
    matched_three_value_column,
):
    baseline_csv = MATCHED_THREE_EXPORTS / "run_metrics.csv"
    if not baseline_csv.is_file():
        raise FileNotFoundError(
            "Matched-three UNSORTED baseline is required but was not found: "
            f"{baseline_csv}. Pass reuse_matched_three_baseline=False to "
            "use the all-feature rows from the ablation CSV instead."
        )

    baseline = pd.read_csv(baseline_csv)
    required = {"pipeline", "seed", matched_three_value_column}
    missing = required - set(baseline)
    if missing:
        raise ValueError(
            f"{baseline_csv} is missing columns: {', '.join(sorted(missing))}"
        )
    baseline = baseline.loc[baseline["pipeline"].eq(REFERENCE_PIPELINE)].copy()
    baseline["seed"] = _require_seed_coverage(
        baseline,
        context=f"{baseline_csv} {REFERENCE_PIPELINE} baseline",
    )
    baseline_values = pd.to_numeric(
        baseline[matched_three_value_column], errors="raise"
    )
    if not np.isfinite(baseline_values).all():
        raise ValueError(
            f"{baseline_csv} contains non-finite {matched_three_value_column} values"
        )

    all_mask = ablation["feature_set"].eq("all")
    all_runs = ablation.loc[all_mask].copy()
    all_runs["seed"] = _require_seed_coverage(
        all_runs,
        context="Ablation all-feature baseline",
    )
    values_by_seed = pd.Series(
        baseline_values.to_numpy(), index=baseline["seed"].to_numpy()
    )
    ablation = ablation.copy()
    ablation.loc[all_mask, value_column] = all_runs["seed"].map(values_by_seed)
    return ablation


def _results(ablation, null_medians, *, value_column):
    results = []
    for feature_set, group in ablation.groupby("feature_set", sort=False):
        values = pd.to_numeric(group[value_column], errors="raise").dropna()
        if values.empty:
            raise ValueError(
                f"No {value_column} values for feature_set={feature_set!r}"
            )
        observed_median = st.median(values)
        results.append({
            "feature_set": str(feature_set),
            "label": LABELS[str(feature_set)],
            "median_value": observed_median,
            "p_value": permutation_p_value(null_medians, observed_median),
        })
    results.sort(key=lambda row: row["median_value"], reverse=True)
    if not results:
        raise ValueError("The ablation results contain no feature sets")
    return results


def _plot_forest(
    null_sorted,
    ax,
    results,
    *,
    value_header,
    show_ylabel,
    header_y,
    show_legend,
):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_positions = list(range(len(results)))
    xmin = float(min(null_sorted[0], min(
        row["median_value"] for row in results
    ))) - 0.03
    value_text_x = xmin + 0.05
    p_text_x = value_text_x + 0.075
    connector_start_x = p_text_x + 0.075

    for y_position, row in zip(y_positions, results):
        ax.plot(
            [connector_start_x, row["median_value"]],
            [y_position, y_position],
            color="#C9C9C9",
            linewidth=0.8,
            zorder=1,
            solid_capstyle="butt",
        )
        color, marker = ablation_figure_set_look(row["feature_set"])
        ax.scatter(
            row["median_value"],
            y_position,
            s=110,
            color=color,
            marker=marker,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in results], fontsize=12)
    xmax = 1.0
    ax.set_ylim(len(results) - 0.5, -1.3)
    if show_ylabel:
        ax.set_ylabel("Feature set", fontsize=14)
    for y_position, row in zip(y_positions, results):
        ax.text(
            value_text_x,
            y_position,
            f"{row['median_value']:.3f}",
            va="center",
            ha="left",
            fontsize=12
        )
        ax.text(
            p_text_x,
            y_position,
            f"{row['p_value']:.3f}",
            va="center",
            ha="left",
            fontsize=12
        )
    ax.text(
        value_text_x,
        header_y,
        f"Median\n{value_header}",
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    ax.text(
        p_text_x, header_y, "p", va="bottom", ha="left", fontweight="bold", fontsize=12,
    )
    ax.set_xlim(xmin, xmax)

    if show_legend:
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], marker=MARKER_ALL, linestyle="none",
                    markersize=9, markerfacecolor=COLOR_ALL,
                    markeredgecolor="white", label="All features",
                ),
                Line2D(
                    [0], [0], marker=MARKER_DROP, linestyle="none",
                    markersize=9, markerfacecolor=COLOR_DROP,
                    markeredgecolor="white", label="Drop-one",
                ),
                Line2D(
                    [0], [0], marker=MARKER_ONLY, linestyle="none",
                    markersize=9, markerfacecolor=COLOR_ONLY,
                    markeredgecolor="white", label="Only-one",
                ),
            ],
            frameon=False,
            loc="upper left",
        )


def _plot_null_distribution(null_medians, ax, *, hist_bins, x_label):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.hist(
        null_medians,
        bins=hist_bins,
        color="#B0B0B0",
        edgecolor="#6E6E6E",
        linewidth=0.8,
    )
    ax.set_ylabel("Permutations", fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    corrected_cutoff = np.quantile(null_medians, 1 - FAMILY_ALPHA)
    ax.axvline(corrected_cutoff, linestyle="--", color="black")


def _make_ablation_metric_forest_figure(
    ablation_csv,
    null_csv,
    output_path=None,
    *,
    value_column,
    null_value_column,
    value_header,
    x_label,
    output_filename,
    matched_three_value_column,
    reuse_matched_three_baseline,
    show_ylabel=False,
    header_y=-0.85,
    forest_legend=False,
    hist_bins=15,
):
    ablation, null_medians = _load_inputs(
        ablation_csv,
        null_csv,
        value_column=value_column,
        null_value_column=null_value_column,
    )
    if reuse_matched_three_baseline:
        ablation = _reuse_matched_three_baseline(
            ablation,
            value_column=value_column,
            matched_three_value_column=matched_three_value_column,
        )
    results = _results(
        ablation,
        null_medians,
        value_column=value_column,
    )
    output_path = Path(
        output_path or get_out_path() / output_filename
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = draw_ablation_forest_figure(
        results,
        null_medians,
        value_header=value_header,
        x_label=x_label,
        show_ylabel=show_ylabel,
        header_y=header_y,
        forest_legend=forest_legend,
        hist_bins=hist_bins,
    )
    with paper_style_context():
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return output_path


def draw_ablation_forest_figure(
    results,
    null_medians,
    *,
    value_header="AUC",
    x_label="Median AUC across 25-repeat CV tests",
    show_ylabel=False,
    header_y=-0.85,
    forest_legend=False,
    hist_bins=20,
):
    null_medians = np.asarray(null_medians, dtype=float)
    with paper_style_context():
        figure, (forest_axis, null_axis) = plt.subplots(
            2,
            1,
            figsize=(9.5, 6.3),
            sharex=True,
            gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.06},
        )
        figure.patch.set_facecolor("white")
        _plot_forest(
            np.sort(null_medians),
            forest_axis,
            list(results),
            value_header=value_header,
            show_ylabel=show_ylabel,
            header_y=header_y,
            show_legend=forest_legend,
        )
        _plot_null_distribution(
            null_medians,
            null_axis,
            hist_bins=hist_bins,
            x_label=x_label,
        )
    return figure


def make_ablation_forest_figure(
    ablation_csv,
    null_csv,
    output_path=None,
    *,
    reuse_matched_three_baseline=True,
    **kwargs,
):
    return _make_ablation_metric_forest_figure(
        ablation_csv,
        null_csv,
        output_path,
        value_column="block_auc",
        null_value_column="median_block_auc",
        value_header="AUC",
        x_label="Median AUC across 25-repeat CV tests",
        output_filename="ablation_forest_null.png",
        matched_three_value_column="block_auc",
        reuse_matched_three_baseline=reuse_matched_three_baseline,
        **kwargs,
    )


def make_ablation_accuracy_forest_figure(
    ablation_csv,
    null_csv,
    output_path=None,
    *,
    reuse_matched_three_baseline=True,
    **kwargs,
):
    return _make_ablation_metric_forest_figure(
        ablation_csv,
        null_csv,
        output_path,
        value_column="block_naive_accuracy",
        null_value_column="median_block_accuracy",
        value_header="Accuracy",
        x_label="Median accuracy across 25 CV folds",
        output_filename="ablation_forest_null_accuracy.png",
        matched_three_value_column="block_naive_accuracy",
        reuse_matched_three_baseline=reuse_matched_three_baseline,
        **kwargs,
    )


__all__ = [
    "draw_ablation_forest_figure",
    "make_ablation_accuracy_forest_figure",
    "make_ablation_forest_figure",
]
