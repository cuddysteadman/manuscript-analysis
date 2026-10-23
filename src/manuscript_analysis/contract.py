from __future__ import annotations

from pathlib import Path


CONFIG_RELATIVE_PATH = Path("config/training_config.json")
N_RUNS = 25
N_SOURCE_SESSIONS = 57
N_CTRL_SESSIONS = 19
N_AD_SESSIONS = 31
N_ELIGIBLE = N_CTRL_SESSIONS + N_AD_SESSIONS
N_SPIKES = 500
MIN_CHUNKS = 10
REFERENCE_PIPELINE = "UNSORTED"
PIPELINE_NAMES = ("UNSORTED", "SORTED", "PRUNED")
TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_REMAINDER = 0.20
SPLIT_SALT = 50
NULL_RUN_ID = "corrected-null-v3"
NULL_PERMUTATIONS = 100
NULL_SPLITS = 25

CLASS_FILES = {
    "CTRL-CO": "CTRL-CO-DAY150.pkl",
    "AD-CO": "AD-CO-DAY150.pkl",
}
CLASS_TO_LABEL = {"CTRL-CO": 0, "AD-CO": 1}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}
COHORT_CLASS_COUNTS = {
    CLASS_TO_LABEL["CTRL-CO"]: N_CTRL_SESSIONS,
    CLASS_TO_LABEL["AD-CO"]: N_AD_SESSIONS,
}

DNU_SESSIONS = (
    "Exp35_AD-CO-day150-1_19Apr22",
    "Exp36_AD-CO-day150-1_19Apr24",
    "Exp15_Ctrl-CO-day150-1_17Apr24",
    "EXP74_CTRL-CO-DAY150-14_NOV24",
    "EXP4_CTRL-CO-DAY150-4_NOV24",
    "EXP2_CTRL-CO-DAY150-2_NOV24",
    "EXP86_CTRL-CO-DAY150-20_NOV24",
)

FEATURE_SETS = {
    "all": (
        "Amplitude (uV)", "Trough to Peak (ms)", "Peak A (uV)",
        "Channel ISI (ms)",
    ),
    "drop_amplitude": (
        "Trough to Peak (ms)", "Peak A (uV)", "Channel ISI (ms)",
    ),
    "drop_trough_to_peak": (
        "Amplitude (uV)", "Peak A (uV)", "Channel ISI (ms)",
    ),
    "drop_peak_a": (
        "Amplitude (uV)", "Trough to Peak (ms)", "Channel ISI (ms)",
    ),
    "drop_channel_isi": (
        "Amplitude (uV)", "Trough to Peak (ms)", "Peak A (uV)",
    ),
    "only_amplitude": ("Amplitude (uV)",),
    "only_trough_to_peak": ("Trough to Peak (ms)",),
    "only_peak_a": ("Peak A (uV)",),
    "only_channel_isi": ("Channel ISI (ms)",),
}

TRAINING_ARTIFACTS = (
    "model.keras", "scaler.npz", "history.csv",
    "block_predictions_test.csv", "session_predictions_test.csv",
    "block_predictions_validation.csv",
    "session_predictions_validation.csv", "metrics.json",
    "sessions_used.csv", "run_complete.json",
)
ANALYSIS_RUN_ARTIFACTS = (
    "metrics.json", "history.csv", "sessions_used.csv",
    "block_predictions_validation.csv", "block_predictions_test.csv",
    "session_predictions_validation.csv", "session_predictions_test.csv",
)
ABLATION_ARCHIVE_ARTIFACTS = (
    "run_manifest.csv", "run_metrics.csv",
    "run_summary.csv", "split_pairing_audit.csv",
    "pooled_confusion_matrices.csv", "history.csv", "sessions_used.csv",
    "block_predictions_validation.csv", "block_predictions_test.csv",
    "session_predictions_validation.csv", "session_predictions_test.csv",
    "unsorted_spike_counts.csv", "baseline_split_manifest.csv",
)


def find_project_root(start: str | Path | None = None) -> Path:
    candidates = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    for candidate in candidates:
        if candidate.is_file():
            candidate = candidate.parent
        for directory in (candidate, *candidate.parents):
            if (directory / "pyproject.toml").is_file():
                return directory
    raise FileNotFoundError("Could not find the manuscript-analysis project root")
