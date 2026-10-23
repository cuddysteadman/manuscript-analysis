from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from scipy.stats import mannwhitneyu

from manuscript_analysis.figures.accuracy import significance_label
from manuscript_analysis.figures.export import save_label_variants
from manuscript_analysis.figures.fig_utils import ROOT
from manuscript_analysis.figures.style import paper_style_context


INPUT_ROOT = ROOT / "input"
FEATURE_JITTER_SEED = 2048
JITTER_WIDTH = 0.15

SIGNED_AMPLITUDE = "Amplitude (uV)"
TROUGH_MINUS_PEAK_B = "Trough (uV) - Peak B (uV)"


class OrganoidClass(NamedTuple):
    name: str
    tick_label: str
    path: Path
    amplitude_method: str


# The two MK-8722 pickles store ``Amplitude (uV)`` as a positive magnitude, so
# their signed amplitude is recovered as in ``tables/1_feature_statistics_per_class``.
ORGANOID_CLASSES = (
    OrganoidClass("hCO", "hCO", INPUT_ROOT / "unsorted" / "CTRL-CO-DAY150.pkl", SIGNED_AMPLITUDE),
    OrganoidClass(
        "hCO^APP", "hCO$^{APP}$", INPUT_ROOT / "unsorted" / "AD-CO-DAY150.pkl", SIGNED_AMPLITUDE
    ),
    OrganoidClass(
        "hCO^APP + 10 nM MK-8722",
        "hCO$^{APP}$\n+ 10 nM",
        INPUT_ROOT / "drug_test" / "AD-CO-DAY150-MKAD2-77-10NM-30DAYS.pkl",
        TROUGH_MINUS_PEAK_B,
    ),
    OrganoidClass(
        "hCO^APP + 100 nM MK-8722",
        "hCO$^{APP}$\n+ 100 nM",
        INPUT_ROOT / "drug_test" / "AD-CO-DAY150-MKAD2-77-100NM-30DAYS.pkl",
        TROUGH_MINUS_PEAK_B,
    ),
    OrganoidClass(
        "hCOL^APP", "hCOL$^{APP}$", INPUT_ROOT / "drug_test" / "OL-AD-CO-DAY150.pkl", SIGNED_AMPLITUDE
    ),
)

# Every class is tested against hCO^APP, except hCO^APP itself, which is tested against hCO.
COMPARISONS = (
    ("hCO", "hCO^APP"),
    ("hCO^APP", "hCO^APP + 10 nM MK-8722"),
    ("hCO^APP", "hCO^APP + 100 nM MK-8722"),
    ("hCO^APP", "hCOL^APP"),
)


class Panel(NamedTuple):
    figure_id: str
    feature: str
    slug: str
    display_name: str
    unit_scale: float = 1.0
    yticks: tuple[tuple[float, str], ...] = ()


PANELS = (
    Panel("5A", "Amplitude (uV)", "amplitude", "Amplitude (uV)",
          yticks=((40, "-40"), (80, "-80"), (160, "-160"))),
    Panel("5B", "Peak A (uV)", "peak_a", "Peak A (uV)"),
    Panel("5C", "Channel ISI (ms)", "channel_isi", "Channel ISI (ms)"),
    Panel("5D", "Trough to Peak (ms)", "trough_to_peak", "Trough to Peak (µs)",
          unit_scale=1000.0, yticks=((200, "200"), (400, "400"), (800, "800"))),
)


def signed_amplitude(recording: pd.DataFrame, method: str) -> pd.Series:
    if method == SIGNED_AMPLITUDE:
        return recording["Amplitude (uV)"]
    if method == TROUGH_MINUS_PEAK_B:
        return recording["Trough (uV)"] - recording["Peak B (uV)"]
    raise ValueError(f"Unknown amplitude method: {method}")


def load_session_medians(classes=ORGANOID_CLASSES, panels=PANELS) -> pd.DataFrame:
    features = [panel.feature for panel in panels]
    medians = []
    for organoid in classes:
        print(f"reading {organoid.path.relative_to(ROOT)}")
        recording = pd.read_pickle(organoid.path)
        values = pd.DataFrame({"session": recording["session"]})
        for feature in features:
            values[feature] = (
                signed_amplitude(recording, organoid.amplitude_method)
                if feature == SIGNED_AMPLITUDE
                else recording[feature]
            )
        del recording
        per_session = (
            values.groupby("session")[features]
            .median()
            .reset_index()
            .melt(id_vars="session", var_name="feature", value_name="session_median")
            .dropna(subset=["session_median"])
        )
        per_session.insert(0, "organoid_class", organoid.name)
        medians.append(per_session)
    return pd.concat(medians, ignore_index=True)


def mann_whitney_tests(
    session_medians: pd.DataFrame, panels=PANELS, comparisons=COMPARISONS
) -> pd.DataFrame:
    rows = []
    for panel in panels:
        feature_medians = session_medians.loc[session_medians["feature"].eq(panel.feature)]
        for first, second in comparisons:
            a = feature_medians.loc[feature_medians["organoid_class"].eq(first), "session_median"]
            b = feature_medians.loc[feature_medians["organoid_class"].eq(second), "session_median"]
            if len(a) < 2 or len(b) < 2:
                continue
            result = mannwhitneyu(a, b, alternative="two-sided")
            rows.append({
                "figure_id": panel.figure_id,
                "feature": panel.feature,
                "first": first,
                "second": second,
                "n_first": len(a),
                "n_second": len(b),
                "u_statistic": result.statistic,
                "p_value": result.pvalue,
                "label": significance_label(result.pvalue),
            })
    return pd.DataFrame(rows)


def panel_points(session_medians: pd.DataFrame, panel: Panel, classes=ORGANOID_CLASSES) -> pd.DataFrame:
    data = session_medians.loc[session_medians["feature"].eq(panel.feature)].copy()
    data["class_order"] = data["organoid_class"].map({c.name: i for i, c in enumerate(classes)})
    data = data.sort_values(["class_order", "session"], kind="stable").reset_index(drop=True)
    is_amplitude = panel.feature == SIGNED_AMPLITUDE
    scaled = data["session_median"] * panel.unit_scale
    data["positive_polarity"] = data["session_median"].gt(0) if is_amplitude else False
    data["plotted_value"] = scaled.abs() if is_amplitude else scaled
    rng = np.random.default_rng(FEATURE_JITTER_SEED)
    data["jitter_offset"] = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, len(data))
    data["plot_x"] = data["class_order"] + 1 + data["jitter_offset"]
    return data[[
        "organoid_class", "class_order", "session", "feature", "session_median",
        "positive_polarity", "plotted_value", "jitter_offset", "plot_x",
    ]]


def _draw_bracket(ax, x1, x2, y, height, text, fontsize=7):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", linewidth=0.8)
    ax.text((x1 + x2) / 2, y + height * 1.1, text, ha="center", va="bottom", fontsize=fontsize)


def _assign_bracket_levels(pairs_x):
    levels = []
    for i, (x1, x2) in enumerate(pairs_x):
        level = 0
        for j, (other_x1, other_x2) in enumerate(pairs_x[:i]):
            if not (x2 <= other_x1 or x1 >= other_x2):
                level = max(level, levels[j] + 1)
        levels.append(level)
    return levels


def make_feature_figure(data: pd.DataFrame, panel: Panel, tests: pd.DataFrame, classes=ORGANOID_CLASSES):
    is_amplitude = panel.feature == SIGNED_AMPLITUDE
    values_by_class = [
        data.loc[data["class_order"].eq(i), "plotted_value"].to_numpy() for i in range(len(classes))
    ]

    fig, ax = plt.subplots(figsize=(len(classes) * 1.2 + 1, 4))
    boxplot = ax.boxplot(
        values_by_class,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(marker=""),
        boxprops=dict(linewidth=1),
        whis=(0, 100),
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("lightgray")

    flipped = data["positive_polarity"].to_numpy(dtype=bool)
    for mask, color in ((~flipped, "black"), (flipped, "red")):
        if mask.any():
            ax.scatter(
                data.loc[mask, "plot_x"], data.loc[mask, "plotted_value"],
                s=20, facecolor=color, edgecolor="none", zorder=3, alpha=0.7,
            )

    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels([c.tick_label for c in classes], fontsize=7)
    ax.set_ylabel(
        f"Median |{panel.display_name}|" if is_amplitude else f"Median {panel.display_name}",
        fontsize=7,
    )
    ax.set_title(panel.display_name, fontsize=8)
    ax.set_yscale("log")
    if panel.yticks:
        ticks = [position for position, _ in panel.yticks]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([text for _, text in panel.yticks]))
        ax.yaxis.set_minor_locator(NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    position = {c.name: i + 1 for i, c in enumerate(classes)}
    panel_tests = tests.loc[tests["figure_id"].eq(panel.figure_id)]
    pairs = [(position[row.first], position[row.second]) for row in panel_tests.itertuples()]
    levels = _assign_bracket_levels(pairs)
    y_low, y_top = ax.get_ylim()
    y_range = y_top - y_low
    step = y_range * 0.08
    for (x1, x2), level, label in zip(pairs, levels, panel_tests["label"]):
        _draw_bracket(ax, x1, x2, y_top + y_range * 0.05 + level * step, step * 0.4, label)
    if pairs:
        ax.set_ylim(y_low, y_top + y_range * 0.05 + (max(levels) + 1.5) * step)

    fig.tight_layout()
    return fig


def export_feature_boxplot(
    session_medians: pd.DataFrame, tests: pd.DataFrame, panel: Panel, output_dir
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = panel_points(session_medians, panel)
    stem = f"{panel.figure_id}_{panel.slug}_session_medians"
    csv_path = output_dir / f"{stem}.csv"
    data.to_csv(csv_path, index=False)
    with paper_style_context(), mpl.rc_context({"mathtext.default": "regular"}):
        figure = make_feature_figure(data, panel, tests)
        paths = save_label_variants(figure, output_dir, stem)
        plt.close(figure)
    return data, [*paths, csv_path]
