from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import artifacts, data, io, modeling
from ..contract import (
    ABLATION_ARCHIVE_ARTIFACTS,
    CONFIG_RELATIVE_PATH,
    DNU_SESSIONS,
    FEATURE_SETS,
    LABEL_TO_CLASS,
    N_RUNS,
)


def setup(
    repo_root,
    *,
    n_runs=N_RUNS,
    reuse_baseline=True,
    output_root=None,
):
    root = Path(repo_root)
    input_root = root / "input"
    config_path = root / CONFIG_RELATIVE_PATH
    baseline_root = root / "results" / "matched_three_pipeline_run"
    baseline_runs = baseline_root / "runs" / "unsorted"
    baseline_exports = baseline_root / "analysis_exports"
    output_root = Path(
        output_root or root / "results" / "matched_ablation_run"
    )
    runs_root = output_root / "runs"
    export_dir = output_root / "analysis_exports"
    runs_root.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    data_config, model_config, train_config = data.load_config(config_path)
    features = list(data_config["features"])
    feature_sets = {name: list(values) for name, values in FEATURE_SETS.items()}
    if feature_sets["all"] != features:
        raise AssertionError(
            "Config feature order differs from the canonical ablation definition"
        )
    pipelines, spike_counts = data.prepare_cohort(
        input_root, ("UNSORTED",), data_config
    )
    records = pipelines["UNSORTED"]
    spike_counts.to_csv(export_dir / "unsorted_spike_counts.csv", index=False)

    splits = artifacts.baseline_splits_from_runs(
        baseline_runs, seeds=range(n_runs), pipeline_name="UNSORTED"
    )
    artifacts.validate_role_membership(
        splits, records=records, dnu_sessions=DNU_SESSIONS
    )
    canonical_path = baseline_exports / "pipeline_splits_25_seeds.csv"
    canonical = artifacts.load_canonical_splits(canonical_path, n_runs)
    columns = ["seed", "pipeline", "split", "session_key", "class_label"]
    if set(map(tuple, splits[columns].to_numpy())) != set(
        map(tuple, canonical[columns].to_numpy())
    ):
        raise AssertionError(
            "Saved baseline memberships differ from the canonical split export"
        )
    if set(splits.session_key) != data.eligible_keys(records):
        raise AssertionError(
            "Canonical baseline runs do not cover all eligible recordings"
        )
    splits.to_csv(export_dir / "baseline_split_manifest.csv", index=False)
    return {
        "repo_root": root,
        "records": records,
        "spike_counts": spike_counts,
        "splits": splits,
        "canonical_splits_path": canonical_path,
        "config_path": config_path,
        "unsorted_path": input_root / "unsorted",
        "baseline_runs_root": baseline_runs,
        "output_root": output_root,
        "runs_root": runs_root,
        "export_dir": export_dir,
        "n_runs": int(n_runs),
        "n_spikes": int(data_config["n"]),
        "features": features,
        "feature_sets": feature_sets,
        "model_config": model_config,
        "train_config": train_config,
        "reuse_baseline": bool(reuse_baseline),
        "manifest_rows": [],
    }


def run_seed(state, feature_set, seed):
    selected = list(state["feature_sets"][feature_set])
    if feature_set == "all" and state["reuse_baseline"]:
        run_dir = state["baseline_runs_root"] / f"seed{int(seed)}"
        artifacts.validate_baseline_run(
            run_dir,
            seed=int(seed),
            pipeline_name="UNSORTED",
            expected_features=selected,
            expected_splits=state["splits"],
            dnu_sessions=DNU_SESSIONS,
        )
        baseline_reused = True
    else:
        run_dir = state["runs_root"] / feature_set / f"seed{int(seed)}"
        if not artifacts.run_is_complete(run_dir):
            bundle = data.build_bundle(
                "UNSORTED",
                int(seed),
                records=state["records"],
                splits=state["splits"],
                all_features=state["features"],
                selected_features=selected,
            )
            modeling.train_condition_seed(
                pipeline_name="UNSORTED",
                seed=int(seed),
                run_dir=run_dir,
                bundle=bundle,
                model_config=state["model_config"],
                train_config=state["train_config"],
                n_spikes=int(state["n_spikes"]),
                label_to_class=LABEL_TO_CLASS,
                feature_set=feature_set,
            )
        artifacts.validate_baseline_run(
            run_dir,
            seed=int(seed),
            pipeline_name="UNSORTED",
            expected_features=selected,
            expected_splits=state["splits"],
            dnu_sessions=DNU_SESSIONS,
        )
        baseline_reused = False
    row = {
        "pipeline": "UNSORTED",
        "feature_set": feature_set,
        "seed": int(seed),
        "features": json.dumps(selected),
        "n_features": len(selected),
        "baseline_reused": baseline_reused,
        "artifact_source": str(run_dir.resolve()),
    }
    state["manifest_rows"].append(row)
    return row


def train(state):
    for feature_set in state["feature_sets"]:
        for seed in range(state["n_runs"]):
            run_seed(state, feature_set, seed)
            pd.DataFrame(state["manifest_rows"]).sort_values(
                ["feature_set", "seed"]
            ).to_csv(state["output_root"] / "run_manifest.csv", index=False)
    manifest = pd.DataFrame(state["manifest_rows"]).sort_values(
        ["feature_set", "seed"]
    ).reset_index(drop=True)
    expected = len(state["feature_sets"]) * state["n_runs"]
    if len(manifest) != expected or manifest.duplicated(
        ["feature_set", "seed"]
    ).any():
        raise AssertionError("Ablation run manifest is incomplete")
    return manifest

def analyze(
    *, manifest_path, export_dir, feature_order, n_runs,
    portable_artifact_sources=False,
):
    manifest = pd.read_csv(manifest_path)
    required = {"pipeline", "feature_set", "seed", "features", "n_features", "baseline_reused", "artifact_source"}
    if required - set(manifest) or manifest.duplicated(["feature_set", "seed"]).any() or set(manifest.feature_set) != set(feature_order) or not manifest.groupby("feature_set").seed.nunique().eq(n_runs).all():
        raise AssertionError("Ablation run manifest is incomplete")
    rich, audit, baseline_memberships = [], [], {}
    order = manifest.assign(_order=pd.Categorical(manifest.feature_set, feature_order, ordered=True)).sort_values(["_order", "seed"])
    dnu = {data.session_key(value) for value in DNU_SESSIONS}
    for row in order.to_dict(orient="records"):
        run_dir = Path(row["artifact_source"])
        exported_row = dict(row)
        if portable_artifact_sources:
            exported_row["artifact_source"] = (
                f"runs/{row['feature_set']}/seed{int(row['seed'])}"
            )
        metrics = json.loads((run_dir / "metrics.json").read_text())
        used = pd.read_csv(run_dir / "sessions_used.csv")
        predictions = pd.read_csv(run_dir / "session_predictions_test.csv")
        if int(metrics["seed"]) != int(row["seed"]) or metrics["pipeline"] != row["pipeline"]:
            raise AssertionError(f"Metadata mismatch in {run_dir}")
        features = json.loads(row["features"])
        if int(row["n_features"]) != len(features):
            raise AssertionError(f"Feature-count mismatch in {run_dir}")
        rich.append({**metrics, "feature_set": row["feature_set"], "features": json.dumps(features), "n_features": len(features), "baseline_reused": artifacts.coerce_bool(row["baseline_reused"]), "artifact_source": exported_row["artifact_source"]})
        used["session_key"] = used.session.map(data.session_key)
        expected_test, predicted_test = set(used.loc[used.split == "test", "session"]), set(predictions.session)
        membership = set(map(tuple, used[["split", "session_key", "class_label"]].to_numpy()))
        if row["feature_set"] == "all":
            baseline_memberships[int(row["seed"])] = membership
        audit.append({"feature_set": row["feature_set"], "seed": int(row["seed"]), "membership_matches_all": membership == baseline_memberships[int(row["seed"])], "test_predictions_match_sessions_used": expected_test == predicted_test, "contains_dnu": bool(set(used.session_key) & dnu)})
    metrics = pd.DataFrame(rich).sort_values(["feature_set", "seed"]).reset_index(drop=True)
    audit = pd.DataFrame(audit)
    if not audit.membership_matches_all.all() or not audit.test_predictions_match_sessions_used.all() or audit.contains_dnu.any():
        raise AssertionError("Saved artifact audit failed")
    metrics.to_csv(export_dir / "run_metrics.csv", index=False)
    audit.to_csv(export_dir / "split_pairing_audit.csv", index=False)
    artifacts.copy_csv_artifacts(manifest, output_dir=export_dir, relative_names=["history.csv", "sessions_used.csv", "block_predictions_validation.csv", "block_predictions_test.csv", "session_predictions_validation.csv", "session_predictions_test.csv"])
    pooled = artifacts.pooled_confusions(metrics)
    pooled.to_csv(export_dir / "pooled_confusion_matrices.csv", index=False)
    summary = metrics.groupby("feature_set").agg(n_runs=("seed", "nunique"), block_auc_mean=("block_auc", "mean"), block_auc_sd=("block_auc", "std"), session_auc_mean=("session_auc", "mean"), session_auc_sd=("session_auc", "std"), train_time_total_s=("train_time_s", "sum")).reset_index()
    summary.to_csv(export_dir / "run_summary.csv", index=False)
    return {"manifest": manifest, "metrics": metrics, "audit": audit, "pooled": pooled, "summary": summary}


def finalize(
    *, manifest_path, output_dir, archive_path, canonical_splits_path,
    config_path, unsorted_path, feature_order=tuple(FEATURE_SETS),
    n_runs=N_RUNS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = analyze(
        manifest_path=manifest_path,
        export_dir=output_dir,
        feature_order=feature_order,
        n_runs=n_runs,
        portable_artifact_sources=True,
    )
    manifest = results["manifest"].copy()
    manifest["artifact_source"] = manifest.apply(
        lambda row: f"runs/{row.feature_set}/seed{int(row.seed)}", axis=1
    )
    manifest.sort_values(["feature_set", "seed"]).to_csv(
        output_dir / "run_manifest.csv", index=False
    )

    canonical = artifacts.load_canonical_splits(canonical_splits_path, n_runs)
    artifacts.validate_role_membership(canonical, dnu_sessions=DNU_SESSIONS)
    canonical.sort_values(
        ["seed", "class_label", "split", "session_key"]
    ).to_csv(output_dir / "baseline_split_manifest.csv", index=False)

    data_config, _, _ = data.load_config(config_path)
    pipelines, spike_counts = data.prepare_cohort(
        Path(unsorted_path).parent,
        ("UNSORTED",),
        data_config,
        storage={"UNSORTED": "none"},
    )
    if set(canonical.session_key) != data.eligible_keys(pipelines["UNSORTED"]):
        raise AssertionError(
            "Canonical ablation splits do not cover exactly the eligible UNSORTED recordings"
        )
    spike_counts.to_csv(output_dir / "unsorted_spike_counts.csv", index=False)

    paths = [output_dir / name for name in ABLATION_ARCHIVE_ARTIFACTS]
    if missing := [path.name for path in paths if not path.exists()]:
        raise FileNotFoundError(f"Missing ablation analysis exports: {missing}")
    io.write_zip(paths, archive_path)


__all__ = [
    "analyze", "finalize", "run_seed", "setup", "train",
]
