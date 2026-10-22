from __future__ import annotations

import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import data, io, modeling
from ..contract import (
    COHORT_CLASS_COUNTS,
    CONFIG_RELATIVE_PATH,
    DNU_SESSIONS,
    MIN_CHUNKS,
    N_AD_SESSIONS,
    N_CTRL_SESSIONS,
    N_ELIGIBLE,
    N_SOURCE_SESSIONS,
    N_SPIKES,
    NULL_PERMUTATIONS,
    NULL_RUN_ID,
    NULL_SPLITS,
    PIPELINE_NAMES,
    REFERENCE_PIPELINE,
    SPLIT_SALT,
    TEST_FRACTION,
    VALIDATION_FRACTION_OF_REMAINDER,
    find_project_root,
)


def _null_permutation_maps(splits, permutation_seeds, split_seeds):
    rows = []
    for split_seed in map(int, split_seeds):
        fold = splits.loc[splits.split_seed.eq(split_seed)].copy()
        fit = fold.loc[fold.split.isin(["train", "val"])].sort_values("session_key")
        keys, labels = fit.session_key.to_numpy(), fit.class_label.to_numpy(int)
        roles = {
            role: set(fit.loc[fit.split.eq(role), "session_key"])
            for role in ("train", "val")
        }
        for perm_seed in map(int, permutation_seeds):
            for attempt in range(10_000):
                seed = np.random.SeedSequence(
                    [SPLIT_SALT, perm_seed, split_seed, attempt]
                )
                shuffled = labels.copy()
                np.random.default_rng(seed).shuffle(shuffled)
                lookup = dict(zip(keys, shuffled))
                if all(len({lookup[key] for key in role_keys}) == 2 for role_keys in roles.values()):
                    break
            else:
                raise RuntimeError(
                    f"No scoreable train/validation label shuffle for permutation={perm_seed}, split={split_seed}"
                )
            assignments = fold[["session", "session_key", "class_name", "class_label", "split"]].copy()
            assignments.insert(0, "split_seed", split_seed)
            assignments.insert(0, "perm_seed", perm_seed)
            assignments["permutation_attempt"] = attempt
            assignments["permuted_label"] = assignments.session_key.map(lookup).astype("Int64")
            assignments.loc[assignments.split.eq("test"), "permuted_label"] = pd.NA
            rows.extend(assignments.to_dict("records"))
    return pd.DataFrame(rows)


def setup(
    base_dir,
    output_dir,
    *,
    permutation_seeds=range(NULL_PERMUTATIONS),
    split_seeds=range(NULL_SPLITS),
    cache_dir=None,
    config_path=None,
):
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    permutation_seeds, split_seeds = tuple(permutation_seeds), tuple(split_seeds)
    if split_seeds != tuple(range(len(split_seeds))):
        raise ValueError("setup split_seeds must be contiguous from zero")
    config_path = Path(
        config_path
        or find_project_root(base_dir) / CONFIG_RELATIVE_PATH
    )
    config = json.loads(config_path.read_text())
    data_config = config["DATA_CFG"]
    model = config["UNSORTED_ARCH_LSTM_CFG"]
    train = config["TRAIN_CFG"]
    if int(data_config["n"]) != N_SPIKES or int(data_config["min_chunks"]) != MIN_CHUNKS:
        raise AssertionError(
            f"Corrected null requires {N_SPIKES}-spike blocks and a {MIN_CHUNKS}-block minimum"
        )
    cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="corrected-null-v3-"))
    pipeline_data, manifests = data.prepare_cohort(
        base_dir,
        PIPELINE_NAMES,
        data_config,
        storage={
            "UNSORTED": "disk",
            "SORTED": "none",
            "PRUNED": "none",
        },
        cache_dir=cache_dir,
    )
    master, availability = data.build_master_manifest(
        pipeline_data, reference_pipeline=REFERENCE_PIPELINE, dnu_sessions=DNU_SESSIONS
    )
    dnu_keys = {data.session_key(value) for value in DNU_SESSIONS}
    if len(pipeline_data["UNSORTED"]) != N_SOURCE_SESSIONS or not dnu_keys <= set(pipeline_data["UNSORTED"]):
        raise AssertionError(
            f"UNSORTED source must contain {N_SOURCE_SESSIONS} sessions including all {len(DNU_SESSIONS)} DNU sessions"
        )
    if master.groupby("class_label").size().to_dict() != COHORT_CLASS_COUNTS:
        raise AssertionError(
            f"Corrected null cohort must contain {N_CTRL_SESSIONS} CTRL and {N_AD_SESSIONS} AD sessions"
        )
    _, all_splits, _, audit = data.create_split_manifests(
        master, pipeline_names=list(PIPELINE_NAMES), n_runs=len(split_seeds),
        test_fraction=TEST_FRACTION,
        validation_fraction_of_remainder=VALIDATION_FRACTION_OF_REMAINDER,
        reference_pipeline=REFERENCE_PIPELINE, split_salt=SPLIT_SALT,
    )
    splits = all_splits.loc[all_splits.pipeline.eq("UNSORTED")].rename(
        columns={"seed": "split_seed"}
    ).reset_index(drop=True)
    expected = {(0, "train"): 12, (0, "val"): 3, (0, "test"): 4,
                (1, "train"): 19, (1, "val"): 5, (1, "test"): 7}
    observed = splits.groupby(["class_label", "split"]).size().div(len(split_seeds)).astype(int).to_dict()
    if observed != expected or not splits.loc[splits.split.eq("test"), "eligible_all_pipelines"].all():
        raise AssertionError(f"Corrected null split contract failed: {observed}")
    maps = _null_permutation_maps(splits, permutation_seeds, split_seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    io.atomic_json(output_dir / "run_manifest.json", {
        "run_id": NULL_RUN_ID, "n_sessions": N_ELIGIBLE,
        "class_counts": {str(label): count for label, count in COHORT_CLASS_COUNTS.items()},
        "split_salt": SPLIT_SALT, "min_chunks": int(data_config["min_chunks"]),
        "permutation_seeds": list(permutation_seeds), "split_seeds": list(split_seeds),
        "config": config,
    })
    io.atomic_csv(splits, output_dir / "split_manifest.csv")
    io.atomic_csv(maps, output_dir / "global_permutation_maps.csv")
    return {
        "output_dir": output_dir, "cache_dir": cache_dir, "records": pipeline_data["UNSORTED"],
        "splits": splits, "maps": maps,
        "features": tuple(data_config["features"]),
        "n_spikes": int(data_config["n"]),
        "model_config": model, "train_config": train,
        "manifests": manifests, "availability": availability,
        "audit": audit,
    }


def _null_fold_dir(output_dir, perm_seed, split_seed):
    return Path(output_dir) / "progress" / f"perm{int(perm_seed)}_split{int(split_seed)}"


def _null_fold_complete(directory, perm_seed, split_seed):
    try:
        marker = json.loads((directory / "complete.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        pd.read_csv(directory / "session_predictions.csv.gz", nrows=1)
        pd.read_csv(directory / "label_assignments.csv.gz", nrows=1)
        return marker["run_id"] == NULL_RUN_ID and (
            int(metrics["perm_seed"]), int(metrics["split_seed"])
        ) == (int(perm_seed), int(split_seed))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_folds(state, *, permutation_seeds, split_seeds, force=False, on_fold_complete=None, verbose=0):
    output_dir, completed = state["output_dir"], []
    map_groups = {
        (int(perm_seed), int(split_seed)): dict(zip(group.session_key, group.permuted_label.astype(int)))
        for (perm_seed, split_seed), group in state["maps"].loc[
            state["maps"].split.isin(["train", "val"])
        ].groupby(["perm_seed", "split_seed"])
    }
    for split_seed in map(int, split_seeds):
        bundle = data.build_bundle(
            "UNSORTED", split_seed, records=state["records"], splits=state["splits"],
            all_features=state["features"], selected_features=state["features"],
        )
        scaler = data.fit_scaler(bundle["X_train"])
        X_train, X_val, X_test = (
            data.apply_scaler(bundle[f"X_{role}"], *scaler) for role in ("train", "val", "test")
        )
        for perm_seed in map(int, permutation_seeds):
            directory = _null_fold_dir(output_dir, perm_seed, split_seed)
            if not force and _null_fold_complete(directory, perm_seed, split_seed):
                continue
            lookup = map_groups[(perm_seed, split_seed)]
            remap = lambda ids: np.asarray([lookup[data.session_key(value)] for value in ids], dtype=np.int8)
            y_train, y_val = (remap(bundle[f"{role}_rec_ids"]) for role in ("train", "val"))
            y_test = bundle["y_test"].astype(np.int8, copy=False)
            if np.unique(y_test).size != 2:
                raise AssertionError("Every true-label test fold must contain both classes")
            weights = data.recording_balanced_weights(y_train, bundle["train_rec_ids"])
            model, history, train_time = modeling.fit_lstm_model(
                X_train, y_train, X_val, y_val, weights,
                input_shape=(state["n_spikes"], len(state["features"])),
                model_config=state["model_config"], train_config=state["train_config"],
                seed=split_seed, verbose=verbose,
            )
            probabilities = model.predict(X_test, verbose=0).ravel()
            rec_ids, rec_p, rec_y, rec_counts = modeling.aggregate_recordings(
                bundle["test_rec_ids"], probabilities, y_test
            )
            session_predictions = pd.DataFrame({
                "perm_seed": perm_seed, "split_seed": split_seed, "session": rec_ids,
                "session_key": [data.session_key(value) for value in rec_ids],
                "class_label": rec_y, "probability_ad": rec_p,
                "predicted_label": (rec_p >= .5).astype(int), "n_blocks": rec_counts,
            })
            assignments = state["maps"].loc[
                state["maps"].perm_seed.eq(perm_seed) & state["maps"].split_seed.eq(split_seed)
            ].copy()
            val_auc = np.asarray(history.history.get("val_auc", [np.nan]), float)
            metrics = {
                "perm_seed": perm_seed, "split_seed": split_seed,
                "n_train_blocks": len(y_train), "n_val_blocks": len(y_val), "n_test_blocks": len(y_test),
                "n_train_sessions": bundle["used_sessions"].split.eq("train").sum(),
                "n_val_sessions": bundle["used_sessions"].split.eq("val").sum(),
                "n_test_sessions": len(rec_ids), "epochs_run": len(history.epoch),
                "best_epoch": int(np.nanargmax(val_auc) + 1) if np.isfinite(val_auc).any() else None,
                "best_val_auc": float(np.nanmax(val_auc)) if np.isfinite(val_auc).any() else None,
                "train_time_s": train_time,
                **modeling.probability_metrics(y_test, probabilities, "block"),
                **modeling.classification_metrics(y_test, probabilities, "block"),
                **modeling.probability_metrics(rec_y, rec_p, "session"),
                **modeling.classification_metrics(rec_y, rec_p, "session"),
            }
            directory.mkdir(parents=True, exist_ok=True)
            io.atomic_json(directory / "metrics.json", metrics)
            io.atomic_csv(session_predictions, directory / "session_predictions.csv.gz")
            io.atomic_csv(assignments, directory / "label_assignments.csv.gz")
            io.atomic_json(directory / "complete.json", {
                "run_id": NULL_RUN_ID, "perm_seed": perm_seed, "split_seed": split_seed
            })
            completed.append((perm_seed, split_seed))
            if on_fold_complete:
                on_fold_complete(directory, perm_seed, split_seed)
            del model
            tf.keras.backend.clear_session(); gc.collect()
        del bundle, X_train, X_val, X_test
        gc.collect()
    return completed


def _corrected_pvalue(values, observed):
    values = pd.Series(values, dtype=float).dropna()
    return None if observed is None or values.empty else float((1 + values.ge(float(observed)).sum()) / (len(values) + 1))


def consolidate(output_dir, *, permutation_seeds, split_seeds, observed_block_auc=None, observed_session_auc=None):
    output_dir = Path(output_dir)
    metrics, predictions, assignments = [], [], []
    for perm_seed in map(int, permutation_seeds):
        for split_seed in map(int, split_seeds):
            directory = _null_fold_dir(output_dir, perm_seed, split_seed)
            if not _null_fold_complete(directory, perm_seed, split_seed):
                raise FileNotFoundError(f"Incomplete null fold: {directory}")
            metrics.append(json.loads((directory / "metrics.json").read_text()))
            predictions.append(pd.read_csv(directory / "session_predictions.csv.gz"))
            assignments.append(pd.read_csv(directory / "label_assignments.csv.gz"))
    scores = pd.DataFrame(metrics).sort_values(["perm_seed", "split_seed"]).reset_index(drop=True)
    session_predictions = pd.concat(predictions, ignore_index=True).sort_values(
        ["perm_seed", "split_seed", "session_key"]
    ).reset_index(drop=True)
    label_assignments = pd.concat(assignments, ignore_index=True).sort_values(
        ["perm_seed", "split_seed", "split", "session_key"]
    ).reset_index(drop=True)
    distribution = scores.groupby("perm_seed", sort=True).agg(
        n_folds=("split_seed", "size"), median_block_auc=("block_auc", "median"),
        mean_block_auc=("block_auc", "mean"), median_session_auc=("session_auc", "median"),
        mean_session_auc=("session_auc", "mean"), median_block_accuracy=("block_accuracy", "median"),
        median_block_balanced_accuracy=("block_balanced_accuracy", "median"),
        median_session_accuracy=("session_accuracy", "median"),
        median_session_balanced_accuracy=("session_balanced_accuracy", "median"),
        total_train_time_s=("train_time_s", "sum"),
    ).reset_index()
    summary = {
        "run_id": NULL_RUN_ID, "n_permutations": len(distribution),
        "n_folds_per_permutation": len(tuple(split_seeds)),
        "observed_block_auc": observed_block_auc, "observed_session_auc": observed_session_auc,
        "null_block_auc_mean": distribution.median_block_auc.mean(),
        "null_block_auc_sd": distribution.median_block_auc.std(ddof=1),
        "null_session_auc_mean": distribution.median_session_auc.mean(),
        "null_session_auc_sd": distribution.median_session_auc.std(ddof=1),
        "p_ge_observed_block_auc": _corrected_pvalue(distribution.median_block_auc, observed_block_auc),
        "p_ge_observed_session_auc": _corrected_pvalue(distribution.median_session_auc, observed_session_auc),
        "minimum_reportable_p": 1 / (len(distribution) + 1),
    }
    io.atomic_csv(scores, output_dir / "null_fold_scores.csv")
    io.atomic_csv(distribution, output_dir / "null_distribution.csv")
    io.atomic_csv(session_predictions, output_dir / "session_predictions.csv.gz")
    io.atomic_csv(label_assignments, output_dir / "label_assignments.csv.gz")
    io.atomic_json(output_dir / "null_summary.json", summary)
    return {"fold_scores": scores, "null_distribution": distribution, "summary": summary}


def cleanup(state):
    shutil.rmtree(state["cache_dir"], ignore_errors=True)


__all__ = [
    "cleanup", "consolidate", "run_folds", "setup",
]
