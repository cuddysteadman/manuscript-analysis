from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manuscript_analysis.figures.export import render_label_variants
from manuscript_analysis.figures.fig_utils import (
    APP_COLOUR_HIST,
    CONTROL_COLOUR,
    HISTORICAL_THRESHOLD,
)
from manuscript_analysis.figures.style import apply_paper_style


EDGES = np.linspace(0, 1, 21)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
BIN_WIDTH = EDGES[1] - EDGES[0]
ALPHA = 0.50
THRESHOLD = round(HISTORICAL_THRESHOLD, 4)

BARE_MARGINS = dict(left=0.02, right=0.99, bottom=0.05, top=0.98)
ROW_MARGINS = {
    True: dict(left=0.08, right=0.99, bottom=0.18, top=0.90, wspace=0.42),
    False: dict(BARE_MARGINS, wspace=0.12),
}


def check_scores(*score_arrays):
    if any(values.size == 0 for values in score_arrays):
        raise ValueError("At least one histogram has no prediction scores.")
    if any(((values < 0) | (values > 1)).any() for values in score_arrays):
        raise ValueError("Prediction scores must be within [0, 1].")


def _finish_axis(ax, include_labels, xlabel, letter=None):
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    if not include_labels:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Session count")
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="both", colors="black", length=3.5, width=0.8)
    if letter is not None:
        ax.text(-0.06, 1.045, letter, transform=ax.transAxes, fontsize=18,
                fontweight="bold", va="bottom", color="black")


def threshold_histogram(ax, include_labels, *, values, letter):
    counts, _ = np.histogram(values, bins=EDGES)
    colors = np.where(CENTRES < THRESHOLD, CONTROL_COLOUR, APP_COLOUR_HIST)
    ax.bar(CENTRES, counts, width=BIN_WIDTH, color=colors, alpha=ALPHA, edgecolor="none")
    _finish_axis(ax, include_labels, r"Prediction score ($\hat{y}$)", letter)


def grouped_histogram(ax, include_labels, *, groups):
    for offset, (values, colour) in zip((-BIN_WIDTH * 0.23, BIN_WIDTH * 0.23), groups):
        counts, _ = np.histogram(values, bins=EDGES)
        ax.bar(CENTRES + offset, counts, width=BIN_WIDTH * 0.46, color=colour,
               alpha=ALPHA, edgecolor="none", linewidth=0)
    _finish_axis(ax, include_labels, r"Classification score ($\hat{y}$)")


def histogram_figure(draws, include_labels, *, figsize):
    apply_paper_style()
    fig, axes = plt.subplots(1, len(draws), figsize=figsize, sharex=len(draws) > 1, squeeze=False)
    for ax, draw in zip(axes.flat, draws):
        draw(ax, include_labels)
    if len(draws) > 1:
        fig.subplots_adjust(**ROW_MARGINS[include_labels])
    elif include_labels:
        fig.tight_layout()
    else:
        fig.subplots_adjust(**BARE_MARGINS)
    return fig


def split_panels(scores, panels):
    missing = set(panels.values()) - set(scores["condition"].dropna())
    if missing:
        raise ValueError(f"Missing conditions: {sorted(missing)}")
    panel_data = {}
    for letter, condition in panels.items():
        frame = scores.loc[scores["condition"].eq(condition)].reset_index(drop=True)
        frame.insert(0, "panel", letter)
        panel_data[letter] = frame
    return panel_data


def export_panel_histograms(panel_data, output_dir, figure_name):
    draws = {
        letter: partial(threshold_histogram, values=frame["probability_ad"].to_numpy(dtype=float), letter=letter)
        for letter, frame in panel_data.items()
    }
    check_scores(*(draw.keywords["values"] for draw in draws.values()))

    csv_paths = []
    for letter, frame in panel_data.items():
        (output_dir / f"({letter})").mkdir(parents=True, exist_ok=True)
        csv_paths.append(output_dir / f"({letter})" / f"{figure_name}_{letter}.csv")
        frame.to_csv(csv_paths[-1], index=False)
    csv_paths.append(output_dir / f"{figure_name}.csv")
    pd.concat(panel_data.values(), ignore_index=True).to_csv(csv_paths[-1], index=False)

    figure_paths = render_label_variants(
        lambda labels: histogram_figure(list(draws.values()), labels, figsize=(10.4, 3.45)),
        output_dir / "sidebyside", figure_name,
    )
    for letter, draw in draws.items():
        figure_paths += render_label_variants(
            lambda labels: histogram_figure([draw], labels, figsize=(3.45, 3.45)),
            output_dir / f"({letter})", f"{figure_name}_{letter}",
        )
    return [*csv_paths, *figure_paths]
