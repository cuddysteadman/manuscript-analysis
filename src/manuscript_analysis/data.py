from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from .contract import (
    CLASS_FILES,
    CLASS_TO_LABEL,
    DNU_SESSIONS,
    N_ELIGIBLE,
    N_SPIKES,
    REFERENCE_PIPELINE,
)

def canonical_session_name(value):
    name = Path(str(value).strip()).name
    return name[:-4] if name.lower().endswith(".dat") else name


def session_key(value):
    return canonical_session_name(value).casefold()


def load_config(path):
    config = json.loads(Path(path).read_text())
    return (
        dict(config["DATA_CFG"]),
        dict(config["UNSORTED_ARCH_LSTM_CFG"]),
        dict(config["TRAIN_CFG"]),
    )


def make_blocks(recording_df, *, features, block_size=N_SPIKES):
    features = list(features)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    missing = set(features) - set(recording_df.columns)
    if missing:
        raise KeyError(f"Recording is missing features: {sorted(missing)}")

    values = recording_df.loc[:, features].copy()
    if "Channel ISI (ms)" in features:
        values["Channel ISI (ms)"] = np.log(values["Channel ISI (ms)"] + 1e-6)
    values = values.to_numpy(dtype=np.float32)
    n_blocks = len(values) // block_size
    if n_blocks == 0:
        raise ValueError(f"Recording has fewer than {block_size} events")
    return np.ascontiguousarray(
        values[: n_blocks * block_size].reshape(n_blocks, block_size, -1)
    )


def prepare_inputs(
    input_root,
    pipelines,
    data_config,
    *,
    storage=None,
    cache_dir=None,
):
    input_root, storage = Path(input_root), storage or {}
    records, counts = {}, []
    for name in pipelines:
        pipeline_records, pipeline_counts = prepare_pipeline(
            name,
            input_root / name.lower(),
            class_files=CLASS_FILES,
            class_to_label=CLASS_TO_LABEL,
            features=data_config["features"],
            n_spikes=int(data_config["n"]),
            min_chunks=int(data_config["min_chunks"]),
            max_chunks=int(data_config["max_chunks"]),
            cutoff_time=data_config.get("cutoff_time"),
            dnu_sessions=DNU_SESSIONS,
            block_storage=storage.get(name, "memory"),
            block_cache_dir=cache_dir,
        )
        records[name] = pipeline_records
        counts.append(pipeline_counts)
    return records, pd.concat(counts, ignore_index=True)


def eligible_keys(records):
    return {key for key, record in records.items() if record["eligible"]}


def prepare_cohort(
    input_root, pipelines, data_config, *, storage=None, cache_dir=None
):
    records, counts = prepare_inputs(
        input_root, pipelines, data_config, storage=storage, cache_dir=cache_dir
    )
    n_eligible = len(eligible_keys(records[REFERENCE_PIPELINE]))
    if n_eligible != N_ELIGIBLE:
        raise AssertionError(
            f"Cohort must contain exactly {N_ELIGIBLE} eligible "
            f"{REFERENCE_PIPELINE} recordings, found {n_eligible}"
        )
    return records, counts


def _class_path(folder, class_name, class_files):
    path = Path(folder) / class_files[class_name]
    if path.exists():
        return path
    matches = sorted(Path(folder).glob(f"*{class_name}*.pkl"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve {class_name} under {folder}")


def prepare_pipeline(
    pipeline_name, pipeline_path, *, class_files, class_to_label, features,
    n_spikes, min_chunks, max_chunks, cutoff_time, dnu_sessions=DNU_SESSIONS,
    block_storage="memory", block_cache_dir=None,
):
    if block_storage not in {"memory", "disk", "none"}:
        raise ValueError("block_storage must be 'memory', 'disk', or 'none'")
    if block_storage == "disk":
        block_cache_dir = Path(block_cache_dir)
        block_cache_dir.mkdir(parents=True, exist_ok=True)
    features = list(features)
    dnu_keys = {session_key(value) for value in dnu_sessions}
    records, rows = {}, []
    for class_name, class_label in class_to_label.items():
        path = _class_path(pipeline_path, class_name, class_files)
        print(f"[{pipeline_name}] reading {path}")
        frame = pd.read_pickle(path)
        missing = {"session", *features} - set(frame.columns)
        if not isinstance(frame, pd.DataFrame) or frame.empty or missing:
            raise ValueError(f"Invalid input {path}; missing={sorted(missing)}")
        raw_counts = frame.groupby("session", sort=False).size().to_dict()
        eligible_sources = {
            source for source, raw_count in raw_counts.items()
            if session_key(source) not in dnu_keys
            and int(raw_count) // n_spikes >= min_chunks
        }
        if cutoff_time is not None:
            if "Sample" not in frame:
                raise KeyError(f"{path} needs Sample for cutoff_time")
            keep = frame["Sample"] < int(cutoff_time)
            after_counts = frame.loc[keep].groupby("session", sort=False).size().to_dict()
            work = (
                frame.loc[frame["session"].isin(eligible_sources) & keep].copy()
                if block_storage != "none" else None
            )
        else:
            after_counts = raw_counts
            work = (
                frame.loc[frame["session"].isin(eligible_sources)].copy()
                if block_storage != "none" else None
            )
        if work is not None and "Channel ISI (ms)" in features:
            work["Channel ISI (ms)"] = np.log(
                work["Channel ISI (ms)"] + 1e-6
            )
        for source, raw_count in raw_counts.items():
            key, session = session_key(source), canonical_session_name(source)
            if key in records:
                raise ValueError(f"Duplicate canonical session: {session}")
            after = int(after_counts.get(source, 0))
            complete = after // n_spikes
            used = min(complete, max_chunks)
            dnu = key in dnu_keys
            required = min_chunks
            eligible = source in eligible_sources
            reason = "DNU" if dnu else (
                f"insufficient_complete_blocks:{complete}<{required}"
                if not eligible else ""
            )
            blocks, blocks_path = None, None
            if eligible and block_storage != "none":
                values = work.loc[work["session"] == source, features].to_numpy(
                    dtype=np.float32, copy=True
                )[: used * n_spikes]
                blocks = values.reshape(used, n_spikes, len(features))
                if block_storage == "disk":
                    safe_key = key.replace("/", "_")
                    blocks_path = block_cache_dir / f"{pipeline_name.lower()}_{safe_key}.npy"
                    np.save(blocks_path, blocks, allow_pickle=False)
                    blocks = None
            record = {
                "session": session, "session_key": key, "class_name": class_name,
                "class_label": int(class_label), "eligible": eligible,
                "omission_reason": reason, "spike_count_raw": int(raw_count),
                "spike_count_after_cutoff": after,
                "complete_blocks": complete,
                "blocks_used": used, "minimum_blocks_required": required,
                "blocks": blocks,
                "blocks_path": None if blocks_path is None else str(blocks_path),
            }
            records[key] = record
            rows.append({"pipeline": pipeline_name, "source_pickle": str(path), **{k: v for k, v in record.items() if k != "blocks"}})
        del frame, work
        gc.collect()
    return records, pd.DataFrame(rows)


def build_master_manifest(pipeline_data, *, reference_pipeline, dnu_sessions=DNU_SESSIONS):
    dnu_keys = {session_key(value) for value in dnu_sessions}
    rows = [
        {"session_key": key, "session": record["session"], "class_name": record["class_name"], "class_label": record["class_label"]}
        for key, record in pipeline_data[reference_pipeline].items() if key not in dnu_keys
    ]
    if not rows:
        raise ValueError("The reference cohort is empty")
    master = pd.DataFrame(rows).sort_values(["class_label", "session_key"]).reset_index(drop=True)
    availability = []
    for pipeline, records in pipeline_data.items():
        suffix = pipeline.lower()
        master[f"eligible_{suffix}"] = master.session_key.map(lambda key: bool(records.get(key, {}).get("eligible")))
        master[f"spikes_{suffix}"] = master.session_key.map(lambda key: records.get(key, {}).get("spike_count_after_cutoff", np.nan))
        master[f"blocks_{suffix}"] = master.session_key.map(lambda key: records.get(key, {}).get("blocks_used", 0))
    eligible_columns = [f"eligible_{name.lower()}" for name in pipeline_data]
    master["eligible_all_pipelines"] = master[eligible_columns].all(axis=1)
    for row in master.itertuples(index=False):
        for pipeline, records in pipeline_data.items():
            record = records.get(row.session_key)
            availability.append({
                "session": row.session, "session_key": row.session_key,
                "class_name": row.class_name, "class_label": row.class_label,
                "pipeline": pipeline, "present": record is not None,
                "eligible": bool(record and record["eligible"]),
                "omission_reason": "missing_from_pipeline" if record is None else record["omission_reason"],
                "spike_count_after_cutoff": np.nan if record is None else record["spike_count_after_cutoff"],
                "complete_blocks": np.nan if record is None else record["complete_blocks"],
                "blocks_used": 0 if record is None else record["blocks_used"],
                "eligible_all_pipelines": row.eligible_all_pipelines,
            })
    return master, pd.DataFrame(availability)


def _quota(n, test_fraction, validation_fraction):
    test = math.ceil(test_fraction * n)
    validation = math.ceil(validation_fraction * (n - test))
    train = n - test - validation
    if min(train, validation, test) <= 0:
        raise ValueError(f"Cannot split {n} eligible recordings into train/val/test")
    return {"eligible": n, "train": train, "val": validation, "test": test}


def create_split_manifests(
    master, *, pipeline_names, n_runs, test_fraction,
    validation_fraction_of_remainder, reference_pipeline, split_salt,
):
    names = list(pipeline_names)
    labels = sorted(master.class_label.unique())
    quotas, quota_rows = {}, []
    for pipeline in names:
        quotas[pipeline] = {}
        column = f"eligible_{pipeline.lower()}"
        for label in labels:
            quota = _quota(int((master[column] & master.class_label.eq(label)).sum()), test_fraction, validation_fraction_of_remainder)
            quotas[pipeline][label] = quota
            quota_rows.append({"pipeline": pipeline, "class_label": label, "class_name": master.loc[master.class_label.eq(label), "class_name"].iloc[0], **quota})
    split_rows, candidate_rows = [], []
    for seed in range(n_runs):
        for label in labels:
            cohort = master.loc[master.class_label.eq(label)]
            common = sorted(cohort.loc[cohort.eligible_all_pipelines, "session_key"])
            order = list(np.random.default_rng(np.random.SeedSequence([split_salt, seed, int(label)])).permutation(common))
            largest = max(quotas[name][label]["test"] for name in names)
            if len(order) < largest:
                raise AssertionError(f"seed={seed}, class={label}: insufficient common test candidates")
            for rank, key in enumerate(order[:largest], 1):
                row = cohort.loc[cohort.session_key.eq(key)].iloc[0]
                candidate_rows.append({"seed": seed, "class_label": label, "class_name": row.class_name, "test_rank": rank, "session": row.session, "session_key": key})
            for index, pipeline in enumerate(names):
                quota = quotas[pipeline][label]
                eligible = set(cohort.loc[cohort[f"eligible_{pipeline.lower()}"], "session_key"])
                test = set(order[:quota["test"]])
                remaining = sorted(eligible - test)
                validation = set(np.random.default_rng(np.random.SeedSequence([split_salt, seed, int(label), index, 1])).permutation(remaining)[:quota["val"]])
                train = eligible - test - validation
                if {"train": len(train), "val": len(validation), "test": len(test)} != {role: quota[role] for role in ("train", "val", "test")}:
                    raise AssertionError(f"Derived quota mismatch for {pipeline}, seed={seed}")
                for row in cohort.itertuples(index=False):
                    role = "test" if row.session_key in test else "val" if row.session_key in validation else "train" if row.session_key in train else "unavailable"
                    split_rows.append({
                        "seed": seed, "pipeline": pipeline, "session": row.session,
                        "session_key": row.session_key, "class_name": row.class_name,
                        "class_label": row.class_label, "split": role,
                        "eligible_in_pipeline": bool(getattr(row, f"eligible_{pipeline.lower()}")),
                        "eligible_all_pipelines": bool(row.eligible_all_pipelines),
                        "used_by_pipeline": role != "unavailable",
                    })
    splits, candidates = pd.DataFrame(split_rows), pd.DataFrame(candidate_rows)
    audit = []
    for seed in range(n_runs):
        reference_test = set(splits.loc[(splits.seed == seed) & (splits.pipeline == reference_pipeline) & (splits.split == "test"), "session_key"])
        for pipeline in names:
            group = splits.loc[(splits.seed == seed) & (splits.pipeline == pipeline)]
            roles = {role: set(group.loc[group.split == role, "session_key"]) for role in ("train", "val", "test")}
            audit.append({
                "seed": seed, "pipeline": pipeline,
                "train_val_overlap": len(roles["train"] & roles["val"]),
                "train_test_overlap": len(roles["train"] & roles["test"]),
                "val_test_overlap": len(roles["val"] & roles["test"]),
                "test_is_subset_of_reference": roles["test"] <= reference_test,
                "all_test_eligible_all_pipelines": bool(group.loc[group.split == "test", "eligible_all_pipelines"].all()),
            })
    audit = pd.DataFrame(audit)
    if not audit[["train_val_overlap", "train_test_overlap", "val_test_overlap"]].eq(0).all().all() or not audit[["test_is_subset_of_reference", "all_test_eligible_all_pipelines"]].all().all():
        raise AssertionError("Split audit failed")
    return pd.DataFrame(quota_rows), splits, candidates, audit


def _role_data(pipeline, seed, role, records, splits, all_features, selected_features):
    seed_column = "seed" if "seed" in splits else "split_seed"
    rows = splits.loc[(splits.pipeline == pipeline) & (splits[seed_column] == seed) & (splits.split == role)].sort_values(["class_label", "session_key"])
    indices = [list(all_features).index(feature) for feature in selected_features]
    if len(indices) != len(selected_features):
        raise KeyError("A selected feature is not present in the source data")
    blocks, labels, recording_ids, used = [], [], [], []
    for row in rows.itertuples(index=False):
        record = records.get(row.session_key)
        if not record or not record["eligible"]:
            raise AssertionError(f"Unavailable {role} session: {row.session}")
        source = record.get("blocks")
        if source is None and record.get("blocks_path"):
            source = np.load(record["blocks_path"], mmap_mode="r", allow_pickle=False)
        if source is None:
            raise AssertionError(f"No blocks materialized for {row.session}")
        value = source[..., indices]
        blocks.append(value)
        labels.append(np.full(len(value), row.class_label, dtype=np.int8))
        recording_ids.append(np.full(len(value), row.session, dtype=object))
        used.append({"pipeline": pipeline, "seed": seed, "split": role, "session": row.session, "session_key": row.session_key, "class_name": row.class_name, "class_label": row.class_label, "n_blocks": len(value)})
    if not blocks:
        raise ValueError(f"{pipeline} seed={seed} has no {role} data")
    return np.concatenate(blocks).astype(np.float32, copy=False), np.concatenate(labels), np.concatenate(recording_ids), pd.DataFrame(used)


def recording_balanced_weights(labels, recording_ids):
    labels, recording_ids = np.asarray(labels), np.asarray(recording_ids)
    weights = np.zeros(len(labels), dtype=np.float32)
    for label in np.unique(labels):
        class_mask = labels == label
        class_records = np.unique(recording_ids[class_mask])
        for record in class_records:
            mask = class_mask & (recording_ids == record)
            weights[mask] = 1.0 / (len(np.unique(labels)) * len(class_records) * mask.sum())
    return weights * (len(weights) / weights.sum())


def build_bundle(pipeline, seed, *, records, splits, all_features, selected_features):
    values = {role: _role_data(pipeline, seed, role, records, splits, all_features, selected_features) for role in ("train", "val", "test")}
    train, validation, test = values["train"], values["val"], values["test"]
    seed_column = "seed" if "seed" in splits else "split_seed"
    expected = set(splits.loc[(splits.pipeline == pipeline) & (splits[seed_column] == seed) & (splits.split == "test"), "session"])
    if set(np.unique(test[2])) != expected:
        raise AssertionError("Test recording IDs drifted from the split manifest")
    return {
        "X_train": train[0], "y_train": train[1], "train_rec_ids": train[2],
        "train_w": recording_balanced_weights(train[1], train[2]),
        "X_val": validation[0], "y_val": validation[1], "val_rec_ids": validation[2],
        "X_test": test[0], "y_test": test[1], "test_rec_ids": test[2],
        "used_sessions": pd.concat([train[3], validation[3], test[3]], ignore_index=True),
        "features": tuple(selected_features),
    }


def fit_scaler(X):
    flat = X.reshape(-1, X.shape[-1])
    lo, hi = np.quantile(flat, .01, axis=0), np.quantile(flat, .99, axis=0)
    clipped = np.clip(flat, lo, hi)
    return tuple(np.asarray(value, dtype=np.float32) for value in (lo, hi, clipped.mean(axis=0), clipped.std(axis=0) + 1e-8))


def apply_scaler(X, lo, hi, mean, standard_deviation):
    return ((np.clip(X, lo, hi) - mean) / standard_deviation).astype(np.float32)
