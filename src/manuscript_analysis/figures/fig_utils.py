from pathlib import Path

from manuscript_analysis import find_project_root

ROOT = find_project_root()
MATCHED_THREE_EXPORTS = ROOT / "results" / "matched_three_pipeline_run" / "analysis_exports"
MATCHED_ABLATION_EXPORTS = ROOT / "results" / "matched_ablation_run" / "analysis_exports"
RECORDING_SCORE_EXPORTS = ROOT / "output" / "test_data"

HISTORICAL_THRESHOLD = 0.3574284

CONTROL_COLOUR = "#7d2e8f"
APP_COLOUR = "#00168f"
APP_COLOUR_7B = "#001694"
APP_COLOUR_HIST = "#00189a"

COLOR_ALL = "#000000"
COLOR_DROP = "#d3503c"
COLOR_ONLY = "#11c6b6"
MARKER_ALL = "o"
MARKER_DROP = "s"
MARKER_ONLY = "^"

LABELS = {
    "all": "All",
    "drop_amplitude": "Drop Amplitude",
    "drop_trough_to_peak": "Drop TPT",
    "drop_peak_a": "Drop Peak A",
    "drop_channel_isi": "Drop cISI",
    "only_amplitude": "Only Amplitude",
    "only_trough_to_peak": "Only TPT",
    "only_peak_a": "Only Peak A",
    "only_channel_isi": "Only cISI",
}


def get_out_path(*parts) -> Path:
    out = ROOT.joinpath("output", "figures", *parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def report_saved(paths) -> None:
    for path in paths:
        print(f"Saved {Path(path).relative_to(ROOT)}")


def ablation_figure_set_look(feature_set: str) -> tuple[str, str]:
    if feature_set == "all":
        return COLOR_ALL, MARKER_ALL
    if feature_set.startswith("drop_"):
        return COLOR_DROP, MARKER_DROP
    if feature_set.startswith("only_"):
        return COLOR_ONLY, MARKER_ONLY
    raise ValueError(f"Unrecognized feature_set: {feature_set!r}")
