from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import io
from .contract import (
    DNU_SESSIONS,
    N_ELIGIBLE,
    REFERENCE_PIPELINE,
    TRAINING_ARTIFACTS,
)
from .data import session_key

def run_is_complete(run_dir):
    return all((Path(run_dir) / name).exists() for name in TRAINING_ARTIFACTS)


def write_sessions_used(frame, run_dir):
    io.atomic_csv(frame, Path(run_dir) / "sessions_used.csv")


def write_training_run(
    run_dir,
    *,
    model,
    scaler,
    features,
    predictions,
    history,
    metrics,
    completion,
):
    run_dir = Path(run_dir)
    for name, frame in predictions.items():
        io.atomic_csv(frame, run_dir / name)
    io.atomic_csv(history, run_dir / "history.csv")
    model.save(run_dir / "model.keras")
    lo, hi, mean, standard_deviation = scaler
    np.savez(
        run_dir / "scaler.npz",
        lo=lo,
        hi=hi,
        mu=mean,
        sd=standard_deviation,
        features=np.asarray(features),
    )
    io.atomic_json(run_dir / "metrics.json", metrics)
    io.atomic_json(run_dir / "run_complete.json", completion)


def coerce_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value.strip().casefold() in {"true", "1", "yes"} if isinstance(value, str) else bool(value)


def baseline_splits_from_runs(runs_root, *, seeds, pipeline_name="UNSORTED"):
    parts = []
    for seed in seeds:
        path = Path(runs_root) / f"seed{seed}" / "sessions_used.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical split artifact: {path}")
        frame = pd.read_csv(path)
        required = {"seed", "pipeline", "split", "session", "class_label"}
        if missing := required - set(frame.columns):
            raise KeyError(f"{path} is missing {sorted(missing)}")
        if set(frame.seed.astype(int)) != {seed} or set(frame.pipeline) != {pipeline_name}:
            raise AssertionError(f"Unexpected seed or pipeline in {path}")
        frame["session_key"] = frame.session.map(session_key)
        if "class_name" not in frame:
            frame["class_name"] = frame.class_label.astype(str)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    validate_role_membership(result)
    return result


def load_canonical_splits(path, n_runs):
    n_runs = int(n_runs)
    splits = pd.read_csv(path)
    required = {"seed", "pipeline", "split", "session", "session_key", "class_label"}
    if required - set(splits.columns):
        raise AssertionError("Canonical split CSV is missing required columns")
    seeds = splits.seed.astype(int)
    splits = splits.loc[
        splits.pipeline.eq(REFERENCE_PIPELINE)
        & splits.split.isin(["train", "val", "test"])
        & seeds.isin(range(n_runs))
    ].assign(seed=seeds)
    if set(splits.seed) != set(range(n_runs)):
        raise AssertionError(
            f"Canonical split CSV does not contain {n_runs} {REFERENCE_PIPELINE} seeds"
        )
    sizes = splits.groupby("seed").size()
    if not (wrong := sizes[sizes.ne(N_ELIGIBLE)]).empty:
        raise AssertionError(
            f"Canonical {REFERENCE_PIPELINE} seeds must each cover {N_ELIGIBLE} "
            f"recordings, found {wrong.to_dict()}"
        )
    return splits


def validate_role_membership(splits, *, records=None, dnu_sessions=DNU_SESSIONS):
    dnu = {session_key(value) for value in dnu_sessions}
    for (pipeline, seed), group in splits.groupby(["pipeline", "seed"]):
        roles = {role: set(group.loc[group.split == role, "session_key"]) for role in ("train", "val", "test")}
        if any(not values for values in roles.values()) or roles["train"] & roles["val"] or roles["train"] & roles["test"] or roles["val"] & roles["test"]:
            raise AssertionError(f"Invalid split roles for {pipeline} seed={seed}")
        used = set.union(*roles.values())
        if used & dnu or (records is not None and any(not records.get(key, {}).get("eligible") for key in used)):
            raise AssertionError(f"Invalid session assignment for {pipeline} seed={seed}")


def validate_baseline_run(run_dir, *, seed, pipeline_name, expected_features, expected_splits, dnu_sessions=DNU_SESSIONS):
    run_dir = Path(run_dir)
    if missing := [name for name in TRAINING_ARTIFACTS if not (run_dir / name).exists()]:
        raise FileNotFoundError(f"Incomplete run {run_dir}; missing {missing}")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if int(metrics["seed"]) != seed or metrics["pipeline"] != pipeline_name or metrics.get("architecture") != "UNSORTED_ARCH":
        raise AssertionError(f"Unexpected model metadata in {run_dir}")
    with np.load(run_dir / "scaler.npz", allow_pickle=True) as scaler:
        if [str(value) for value in scaler["features"]] != list(expected_features):
            raise AssertionError(f"Feature mismatch in {run_dir}")
    observed = pd.read_csv(run_dir / "sessions_used.csv")
    observed["session_key"] = observed.session.map(session_key)
    expected = expected_splits.loc[(expected_splits.pipeline == pipeline_name) & (expected_splits.seed == seed) & expected_splits.split.isin(["train", "val", "test"])]
    columns = ["split", "session_key", "class_label"]
    if set(map(tuple, observed[columns].to_numpy())) != set(map(tuple, expected[columns].to_numpy())) or set(observed.session_key) & {session_key(value) for value in dnu_sessions}:
        raise AssertionError(f"Session membership differs from the manifest in {run_dir}")
    return metrics


def read_run_artifacts(runs_root, *, pipeline_names, seeds):
    files = {
        "run_metrics": "metrics.json", "training_history": "history.csv",
        "block_predictions_test": "block_predictions_test.csv", "session_predictions_test": "session_predictions_test.csv",
        "block_predictions_validation": "block_predictions_validation.csv", "session_predictions_validation": "session_predictions_validation.csv",
        "sessions_used_by_run": "sessions_used.csv",
    }
    collected = {name: [] for name in files}
    for pipeline in pipeline_names:
        for seed in seeds:
            run_dir = Path(runs_root) / pipeline.lower() / f"seed{seed}"
            if not run_is_complete(run_dir):
                raise FileNotFoundError(f"Incomplete run: {run_dir}")
            for name, relative in files.items():
                path = run_dir / relative
                collected[name].append(json.loads(path.read_text()) if relative.endswith(".json") else pd.read_csv(path))
    return {name: pd.DataFrame(parts) if name == "run_metrics" else pd.concat(parts, ignore_index=True) for name, parts in collected.items()}


def pooled_confusions(metrics):
    grouping = ["pipeline"] + (["feature_set"] if "feature_set" in metrics else [])
    rows = []
    for keys, group in metrics.groupby(grouping):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identifiers = dict(zip(grouping, keys))
        for level in ("block", "session"):
            for mode in ("tuned", "naive"):
                prefix = f"{level}_{mode}"
                row = {**identifiers, "level": level, "threshold_mode": mode, **{key: int(group[f"{prefix}_{key}"].sum()) for key in ("tn", "fp", "fn", "tp")}}
                row["n_total"] = sum(row[key] for key in ("tn", "fp", "fn", "tp"))
                rows.append(row)
    return pd.DataFrame(rows)


def paired_pipeline_exports(artifacts, *, pipeline_names):
    from . import modeling

    sessions, blocks, metrics = artifacts["session_predictions_test"], artifacts["block_predictions_test"], artifacts["run_metrics"]
    session_parts, block_parts, metric_rows = [], [], []
    for seed in sorted(metrics.seed.unique()):
        session_seed, block_seed = sessions.loc[sessions.seed == seed], blocks.loc[blocks.seed == seed]
        shared_sessions = set.intersection(*[set(session_seed.loc[session_seed.pipeline == pipeline, "session"]) for pipeline in pipeline_names])
        paired_sessions, paired_blocks = session_seed.loc[session_seed.session.isin(shared_sessions)], block_seed.loc[block_seed.session.isin(shared_sessions)]
        session_parts.append(paired_sessions); block_parts.append(paired_blocks)
        for pipeline in pipeline_names:
            run_metric = metrics.loc[(metrics.pipeline == pipeline) & (metrics.seed == seed)].iloc[0]
            session_group, block_group = paired_sessions.loc[paired_sessions.pipeline == pipeline], paired_blocks.loc[paired_blocks.pipeline == pipeline]
            sy, sp = session_group.class_label.to_numpy(int), session_group.probability_ad.to_numpy(float)
            by, bp = block_group.class_label.to_numpy(int), block_group.probability_ad.to_numpy(float)
            metric_rows.append({
                "pipeline": pipeline, "seed": int(seed), "n_paired_test_sessions": len(session_group), "n_paired_test_ctrl_sessions": int((sy == 0).sum()), "n_paired_test_ad_sessions": int((sy == 1).sum()), "n_paired_test_blocks": len(block_group),
                **modeling.probability_metrics(by, bp, "block"),
                **modeling.probability_metrics(sy, sp, "session"),
                **modeling.classification_metrics(by, bp, "block_tuned", float(run_metric.block_tuned_threshold)),
                **modeling.classification_metrics(by, bp, "block_naive", .5),
                **modeling.classification_metrics(sy, sp, "session_tuned", float(run_metric.session_tuned_threshold)),
                **modeling.classification_metrics(sy, sp, "session_naive", .5),
            })
    return {"paired_session_predictions_test": pd.concat(session_parts, ignore_index=True), "paired_block_predictions_test": pd.concat(block_parts, ignore_index=True), "paired_run_metrics": pd.DataFrame(metric_rows)}


def copy_csv_artifacts(manifest, *, output_dir, relative_names):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for relative in relative_names:
        header = False
        for row in manifest.sort_values(["feature_set", "seed"]).itertuples(index=False):
            path = Path(row.artifact_source) / relative
            if not path.exists():
                raise FileNotFoundError(f"Missing run artifact: {path}")
            frame = pd.read_csv(path)
            if "feature_set" not in frame:
                frame.insert(1, "feature_set", row.feature_set)
            frame.to_csv(output_dir / relative, mode="a" if header else "w", header=not header, index=False)
            header = True
