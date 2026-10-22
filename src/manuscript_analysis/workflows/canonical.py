from __future__ import annotations

from html import escape
from pathlib import Path

from ..contract import (
    CONFIG_RELATIVE_PATH,
    DNU_SESSIONS,
    LABEL_TO_CLASS,
    N_RUNS,
    PIPELINE_NAMES,
    REFERENCE_PIPELINE,
    SPLIT_SALT,
    TEST_FRACTION,
    VALIDATION_FRACTION_OF_REMAINDER,
)

from .. import artifacts, data, io, modeling


class _NotebookStatus:
    def __init__(self, *, total_fits, cached_fits):
        self.total_fits = total_fits
        self.cached_fits = cached_fits
        self.completed_fits = cached_fits
        self._display = None
        self._html = None
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if get_ipython() is not None:
                self._html = HTML
                self._display = display(HTML(self._content()), display_id=True)
        except ImportError:
            pass

    def _content(self, *, seed=None, n_runs=None, pipeline=None, detail=None):
        text = (
            f"<b>Canonical training</b> &middot; "
            f"{self.completed_fits}/{self.total_fits} fits complete<br>"
            f"Resume: {self.cached_fits} cached &middot; "
            f"{self.total_fits - self.completed_fits} remaining"
        )
        if seed is not None:
            text += (
                f"<br>Seed {seed + 1}/{n_runs} &middot; "
                f"<b>{escape(pipeline)}</b> &middot; {escape(detail)}"
            )
        return text

    def update(self, *, seed, n_runs, pipeline, detail):
        if self._display is None:
            return
        self._display.update(self._html(self._content(
            seed=seed, n_runs=n_runs, pipeline=pipeline, detail=detail,
        )))


class _NotebookEpochProgress(modeling.tf.keras.callbacks.Callback):
    def __init__(self, status, *, seed, n_runs, pipeline):
        super().__init__()
        self.status = status
        self.seed = seed
        self.n_runs = n_runs
        self.pipeline = pipeline

    def on_epoch_end(self, epoch, logs=None):
        self.status.update(
            seed=self.seed,
            n_runs=self.n_runs,
            pipeline=self.pipeline,
            detail=f"epoch {epoch + 1}/{self.params['epochs']}",
        )


def setup(repo_root):
    root = Path(repo_root)
    input_root = root / "input"
    pipeline_paths = {name: input_root / name.lower() for name in PIPELINE_NAMES}
    output_root = root / "results" / "matched_three_pipeline_run"
    runs_root = output_root / "runs"
    export_dir = output_root / "analysis_exports"
    for path in (runs_root, export_dir):
        path.mkdir(parents=True, exist_ok=True)
    data_config, model, training = data.load_config(
        root / CONFIG_RELATIVE_PATH
    )
    features = list(data_config["features"])
    snapshot = {
        "n_runs": N_RUNS,
        "pipeline_paths": {
            key: str(value) for key, value in pipeline_paths.items()
        },
        "dnu_sessions": list(DNU_SESSIONS),
        "test_fraction": TEST_FRACTION,
        "validation_fraction_of_remainder":
            VALIDATION_FRACTION_OF_REMAINDER,
        "reference_pipeline": REFERENCE_PIPELINE,
        "split_salt": SPLIT_SALT,
        "data_cfg": data_config,
        "model_cfg": model,
        "train_cfg": training,
    }
    pipeline_data, spike_counts = data.prepare_cohort(
        input_root, PIPELINE_NAMES, data_config
    )
    master, availability = data.build_master_manifest(
        pipeline_data,
        reference_pipeline=REFERENCE_PIPELINE,
        dnu_sessions=DNU_SESSIONS,
    )
    spike_counts.to_csv(export_dir / "spike_counts_long.csv", index=False)
    master.to_csv(export_dir / "session_manifest_master_50.csv", index=False)
    availability.to_csv(export_dir / "pipeline_availability.csv", index=False)

    metadata = (
        spike_counts[
            ["session_key", "session", "class_name", "class_label"]
        ]
        .drop_duplicates("session_key")
        .set_index("session_key")
    )
    wide = spike_counts.pivot_table(
        index="session_key",
        columns="pipeline",
        values="spike_count_after_cutoff",
        aggfunc="first",
    ).rename(
        columns={
            "SORTED": "sorted",
            "UNSORTED": "unsorted",
            "PRUNED": "pruned",
        }
    )
    wide = metadata.join(wide, how="outer").reset_index()
    wide["excluded_dnu"] = wide.session_key.isin(
        {data.session_key(value) for value in DNU_SESSIONS}
    )
    wide["in_master_cohort"] = wide.session_key.isin(set(master.session_key))
    wide["condition"] = wide.class_name.map(
        {
            "CTRL-CO": "CTRL-CO-DAY150",
            "AD-CO": "AD-CO-DAY150",
        }
    )
    wide.to_csv(export_dir / "spike_counts_raw.csv", index=False)
    wide.loc[wide.in_master_cohort].to_csv(
        export_dir / "spike_counts_master_50.csv", index=False
    )

    quotas, splits, candidates, audit = data.create_split_manifests(
        master,
        pipeline_names=list(pipeline_paths),
        n_runs=N_RUNS,
        test_fraction=TEST_FRACTION,
        validation_fraction_of_remainder=VALIDATION_FRACTION_OF_REMAINDER,
        reference_pipeline=REFERENCE_PIPELINE,
        split_salt=SPLIT_SALT,
    )
    quotas.to_csv(export_dir / "pipeline_split_quotas.csv", index=False)
    splits.to_csv(export_dir / "pipeline_splits_25_seeds.csv", index=False)
    candidates.to_csv(
        export_dir / "test_candidate_supersets_25_seeds.csv", index=False
    )
    audit.to_csv(export_dir / "split_audit.csv", index=False)
    splits.loc[~splits.used_by_pipeline].to_csv(
        export_dir / "pipeline_specific_omissions.csv", index=False
    )
    split_counts = (
        splits.loc[splits.used_by_pipeline]
        .groupby(["pipeline", "seed", "class_name", "class_label", "split"])
        .size()
        .rename("recordings")
        .reset_index()
    )
    split_counts.to_csv(
        export_dir / "pipeline_split_counts.csv", index=False
    )
    snapshot.update(
        {
            "observed_master_counts": master.groupby("class_label")
            .size()
            .astype(int)
            .to_dict(),
            "observed_pipeline_quotas": quotas.to_dict(orient="records"),
        }
    )
    io.atomic_json(export_dir / "config_snapshot.json", snapshot)

    return {
        "pipeline_data": pipeline_data,
        "spike_counts": spike_counts,
        "master": master,
        "availability": availability,
        "quotas": quotas,
        "splits": splits,
        "candidates": candidates,
        "split_audit": audit,
        "split_counts": split_counts,
        "export_dir": export_dir,
        "repo_root": root,
        "pipeline_paths": pipeline_paths,
        "runs_root": runs_root,
        "n_runs": N_RUNS,
        "features": features,
        "n_spikes": int(data_config["n"]),
        "model_config": model,
        "train_config": training,
        "label_to_class": LABEL_TO_CLASS,
        "dnu_sessions": DNU_SESSIONS,
    }

def consolidate(state):
    runs_root = state["runs_root"]
    pipeline_paths = state["pipeline_paths"]
    n_runs = state["n_runs"]
    artifact_frames = artifacts.read_run_artifacts(
        runs_root,
        pipeline_names=list(pipeline_paths),
        seeds=range(n_runs),
    )
    sort_columns = {
        "run_metrics": ["pipeline", "seed"], "training_history": ["pipeline", "seed", "epoch"],
        "block_predictions_test": ["pipeline", "seed", "session", "block_index_within_session"],
        "session_predictions_test": ["pipeline", "seed", "session"],
        "block_predictions_validation": ["pipeline", "seed", "session", "block_index_within_session"],
        "session_predictions_validation": ["pipeline", "seed", "session"],
        "sessions_used_by_run": ["pipeline", "seed", "split", "class_label", "session"],
    }
    for name, frame in artifact_frames.items():
        artifact_frames[name] = frame.sort_values(
            sort_columns[name]
        ).reset_index(drop=True)
        artifact_frames[name].to_csv(
            state["export_dir"] / f"{name}.csv", index=False
        )
    metrics = artifact_frames["run_metrics"]
    pooled = artifacts.pooled_confusions(metrics)
    pooled.to_csv(state["export_dir"] / "pooled_confusion_matrices.csv", index=False)
    paired = artifacts.paired_pipeline_exports(
        artifact_frames, pipeline_names=list(pipeline_paths)
    )
    for name, frame in paired.items():
        frame.to_csv(state["export_dir"] / f"{name}.csv", index=False)
    paired_pooled = artifacts.pooled_confusions(paired["paired_run_metrics"])
    paired_pooled.to_csv(state["export_dir"] / "paired_pooled_confusion_matrices.csv", index=False)
    summary = metrics.groupby("pipeline").agg(
        n_runs=("seed", "nunique"), session_auc_mean=("session_auc", "mean"),
        session_auc_sd=("session_auc", "std"), session_accuracy_mean=("session_tuned_accuracy", "mean"),
        block_auc_mean=("block_auc", "mean"), train_time_total_s=("train_time_s", "sum"),
    ).reset_index()
    summary.to_csv(state["export_dir"] / "run_summary.csv", index=False)
    if len(metrics) != len(pipeline_paths) * n_runs:
        raise AssertionError("The consolidated run table is incomplete")
    return {"artifacts": artifact_frames, "run_metrics": metrics, "pooled_confusions": pooled, "paired_pooled_confusions": paired_pooled, "run_summary": summary}

def train(state):
    runs_root = state["runs_root"]
    pipeline_paths = state["pipeline_paths"]
    n_runs = state["n_runs"]
    features = state["features"]
    n_spikes = state["n_spikes"]
    model_config = state["model_config"]
    train_config = state["train_config"]
    label_to_class = state["label_to_class"]
    dnu_sessions = state["dnu_sessions"]

    total_fits = len(pipeline_paths) * n_runs
    cached_fits = sum(
        artifacts.run_is_complete(
            runs_root / pipeline.lower() / f"seed{seed}"
        )
        for seed in range(n_runs)
        for pipeline in pipeline_paths
    )
    status = _NotebookStatus(
        total_fits=total_fits, cached_fits=cached_fits,
    )

    for seed in range(n_runs):
        printed_seed = False
        for pipeline in pipeline_paths:
            run_dir = runs_root / pipeline.lower() / f"seed{seed}"
            if artifacts.run_is_complete(run_dir):
                artifacts.validate_baseline_run(
                    run_dir, seed=seed, pipeline_name=pipeline,
                    expected_features=features, expected_splits=state["splits"],
                    dnu_sessions=dnu_sessions,
                )
                continue
            if not printed_seed:
                print(f"\n[{seed + 1}/{n_runs}] seed {seed}")
                printed_seed = True
            status.update(
                seed=seed,
                n_runs=n_runs,
                pipeline=pipeline,
                detail="training",
            )
            bundle = data.build_bundle(
                pipeline, seed, records=state["pipeline_data"][pipeline],
                splits=state["splits"], all_features=features, selected_features=features,
            )
            metrics = modeling.train_condition_seed(
                pipeline_name=pipeline, seed=seed, run_dir=run_dir, bundle=bundle,
                model_config=model_config, train_config=train_config, n_spikes=n_spikes,
                label_to_class=label_to_class,
                callbacks=[_NotebookEpochProgress(
                    status, seed=seed, n_runs=n_runs, pipeline=pipeline,
                )],
            )
            status.completed_fits += 1
            status.update(
                seed=seed,
                n_runs=n_runs,
                pipeline=pipeline,
                detail="complete",
            )
            print(
                f"  {pipeline:<9} ✓  {metrics['epochs_run']} epochs · "
                f"{metrics['train_time_s']:.0f}s · "
                f"session AUC {metrics['session_auc']:.3f}"
            )
__all__ = ["consolidate", "setup", "train"]
