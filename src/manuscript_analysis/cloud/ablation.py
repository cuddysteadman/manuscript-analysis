from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .. import io
from ..contract import (
    ANALYSIS_RUN_ARTIFACTS,
    CONFIG_RELATIVE_PATH,
    DNU_SESSIONS,
    FEATURE_SETS,
    N_RUNS,
    TRAINING_ARTIFACTS,
)
from .common import (
    array_counts,
    build_worker_revision,
    clients,
    download,
    get_json,
    install_numpy_pickle_compatibility,
    preflight_resources,
    put_json,
    worker_source_files,
)


FEATURE_SET_NAMES = tuple(FEATURE_SETS)
# run_complete.json goes last so a partial upload never looks complete
upload_order = [n for n in TRAINING_ARTIFACTS if n != "run_complete.json"] + ["run_complete.json"]
INPUT_FILES = (
    "unsorted/AD-CO-DAY150.pkl",
    "unsorted/CTRL-CO-DAY150.pkl",
)
DEPLOYMENT_KEY = "control/deployment.json"
SUBMISSION_KEY = "control/latest_submission.json"


def seed_for_index(profile, index):
    profile, index = profile.upper(), int(index)
    if profile == "SMOKE":
        if index != 0:
            raise ValueError("SMOKE trains seed 0 in one job")
        return 0
    if profile == "FULL":
        if not 0 <= index < N_RUNS:
            raise ValueError(f"FULL index must be in [0, {N_RUNS - 1}]")
        return index
    raise ValueError(f"Unknown ablation profile: {profile}")


def _local_inputs(repo_root):
    root = Path(repo_root)
    return (
        root / CONFIG_RELATIVE_PATH,
        root / "results" / "matched_three_pipeline_run" / "analysis_exports"
        / "pipeline_splits_25_seeds.csv",
    )


def preflight(config, repo_root):
    from .. import artifacts

    checked = preflight_resources(
        config, INPUT_FILES, codebuild=True, require_enabled=True
    )
    config_path, splits_path = _local_inputs(repo_root)
    json.loads(config_path.read_text())
    split_rows = len(artifacts.load_canonical_splits(splits_path, N_RUNS))
    return {**checked, "split_rows": split_rows}


def deploy(config, repo_root):
    repo_root = Path(repo_root)
    checked = preflight(config, repo_root)
    config_path, splits_path = _local_inputs(repo_root)
    deployment = build_worker_revision(
        config,
        files=worker_source_files(repo_root),
        archive_label="ablation-worker",
        build_prefix=f"{config['result_prefix']}/control/build",
        tags={"Project": "matched-ablation", "ManagedBy": "run_feature_ablation"},
    )
    identity_payload = {
        "source_digest": deployment["source_digest"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "inputs": checked["inputs"],
    }
    run_id = "matched-ablation-" + hashlib.sha256(
        io.json_bytes(identity_payload)
    ).hexdigest()[:16]
    s3, = clients(config, "s3")
    snapshot_prefix = f"{config['result_prefix']}/control/runs/{run_id}"
    config_key = f"{snapshot_prefix}/training_config.json"
    splits_key = f"{snapshot_prefix}/pipeline_splits_25_seeds.csv"
    s3.upload_file(str(config_path), config["bucket"], config_key)
    s3.upload_file(str(splits_path), config["bucket"], splits_key)
    deployment.update({
        "run_id": run_id,
        "config_key": config_key,
        "splits_key": splits_key,
        "identity": identity_payload,
    })
    put_json(
        s3,
        config["bucket"],
        f"{config['result_prefix']}/{DEPLOYMENT_KEY}",
        deployment,
    )
    return deployment


def submit(config, profile):
    profile = profile.upper()
    if profile not in {"SMOKE", "FULL"}:
        raise ValueError(f"Unknown ablation profile: {profile}")
    s3, batch = clients(config, "s3", "batch")
    deployment = get_json(
        s3,
        config["bucket"],
        f"{config['result_prefix']}/{DEPLOYMENT_KEY}",
    )
    n_runs = 1 if profile == "SMOKE" else N_RUNS
    archive_key = (
        f"{config['result_prefix']}/smoke/ablation_analysis.zip"
        if profile == "SMOKE"
        else f"{config['result_prefix']}/ablation_analysis.zip"
    )
    environment = [
        {"name": "ABLATION_PROFILE", "value": profile},
        {"name": "ABLATION_RUN_ID", "value": deployment["run_id"]},
        {"name": "ABLATION_BUCKET", "value": config["bucket"]},
        {"name": "ABLATION_INPUT_PREFIX", "value": config["input_prefix"]},
        {"name": "ABLATION_RESULT_PREFIX", "value": config["result_prefix"]},
        {"name": "ABLATION_CONFIG_KEY", "value": deployment["config_key"]},
        {"name": "ABLATION_SPLITS_KEY", "value": deployment["splits_key"]},
        {"name": "ABLATION_N_RUNS", "value": str(n_runs)},
        {"name": "ABLATION_ARCHIVE_KEY", "value": archive_key},
    ]
    stamp = int(time.time())
    training_request = dict(
        jobName=f"ablation-{profile.lower()}-{stamp}",
        jobQueue=config["job_queue"],
        jobDefinition=deployment["job_definition_arn"],
        containerOverrides={
            "environment": [
                *environment,
                {"name": "ABLATION_MODE", "value": "TRAIN"},
                {"name": "ABLATION_SEED", "value": "0"},
            ]
        },
    )
    if profile == "FULL":
        training_request["arrayProperties"] = {"size": n_runs}
    training = batch.submit_job(**training_request)
    finalizer = batch.submit_job(
        jobName=f"ablation-finalize-{profile.lower()}-{stamp}",
        jobQueue=config["job_queue"],
        jobDefinition=deployment["job_definition_arn"],
        dependsOn=[{"jobId": training["jobId"]}],
        containerOverrides={
            "environment": [
                *environment,
                {"name": "ABLATION_MODE", "value": "FINALIZE"},
            ]
        },
    )
    submission = {
        "profile": profile,
        "run_id": deployment["run_id"],
        "array_size": n_runs,
        "is_array": profile == "FULL",
        "training_job_id": training["jobId"],
        "finalizer_job_id": finalizer["jobId"],
        "archive_uri": f"s3://{config['bucket']}/{archive_key}",
    }
    put_json(
        s3,
        config["bucket"],
        f"{config['result_prefix']}/{SUBMISSION_KEY}",
        submission,
    )
    return submission


def monitor(config):
    s3, batch = clients(config, "s3", "batch")
    submission = get_json(
        s3,
        config["bucket"],
        f"{config['result_prefix']}/{SUBMISSION_KEY}",
    )
    jobs = batch.describe_jobs(jobs=[
        submission["training_job_id"],
        submission["finalizer_job_id"],
    ])["jobs"]
    statuses = {job["jobId"]: job["status"] for job in jobs}
    result = {
        **submission,
        "training_status": statuses.get(submission["training_job_id"]),
        "finalizer_status": statuses.get(submission["finalizer_job_id"]),
    }
    if submission["is_array"]:
        result["training_counts"] = array_counts(
            batch, submission["training_job_id"]
        )
    return result


def run_action(action, config, *, repo_root):
    action = action.upper()
    if action == "PREFLIGHT":
        return preflight(config, repo_root)
    if action == "DEPLOY":
        return deploy(config, repo_root)
    if action in {"SMOKE", "FULL"}:
        return submit(config, action)
    if action == "MONITOR":
        return monitor(config)
    raise ValueError(f"Unknown Batch action: {action}")


def _completion_identity(run_id, feature_set, seed):
    return {
        "run_id": run_id,
        "feature_set": feature_set,
        "seed": int(seed),
    }


def _remote_run_complete(s3, bucket, prefix, expected, run_files):
    from botocore.exceptions import ClientError

    try:
        observed = get_json(s3, bucket, f"{prefix}/run_complete.json")
        if any(observed.get(key) != value for key, value in expected.items()):
            return False
        for name in run_files:
            s3.head_object(Bucket=bucket, Key=f"{prefix}/{name}")
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {
            "404", "NoSuchKey", "NotFound"
        }:
            return False
        raise


def _run_prefix(result_prefix, run_id, feature_set, seed):
    return f"{result_prefix}/runs/{run_id}/{feature_set}/seed{seed}"


def _worker_context():
    region = os.environ.get("AWS_REGION", "eu-west-1")
    bucket = os.environ["ABLATION_BUCKET"]
    run_id = os.environ["ABLATION_RUN_ID"]
    result_prefix = os.environ["ABLATION_RESULT_PREFIX"]
    return (
        clients({"region": region}, "s3")[0],
        bucket,
        run_id,
        result_prefix,
    )


def _download_worker_inputs(s3, bucket, run_id):
    root = Path("/cache/ablation") / run_id
    config_path = download(
        s3, bucket, os.environ["ABLATION_CONFIG_KEY"],
        root / "training_config.json",
    )
    splits_path = download(
        s3, bucket, os.environ["ABLATION_SPLITS_KEY"],
        root / "pipeline_splits_25_seeds.csv",
    )
    input_root = root / "input"
    for relative in INPUT_FILES:
        download(
            s3,
            bucket,
            f"{os.environ['ABLATION_INPUT_PREFIX']}/{relative}",
            input_root / relative,
        )
    return config_path, splits_path, input_root


def run_training_worker():
    from .. import artifacts, data
    from ..workflows import ablation as workflows

    install_numpy_pickle_compatibility()
    s3, bucket, run_id, result_prefix = _worker_context()
    profile = os.environ["ABLATION_PROFILE"]
    index = int(
        os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", os.environ["ABLATION_SEED"])
    )
    seed = seed_for_index(profile, index)
    incomplete = []
    for feature_set in FEATURE_SET_NAMES:
        remote_prefix = _run_prefix(result_prefix, run_id, feature_set, seed)
        expected = _completion_identity(run_id, feature_set, seed)
        if _remote_run_complete(
            s3, bucket, remote_prefix, expected, TRAINING_ARTIFACTS
        ):
            print(f"[{feature_set}] seed={seed}: complete in S3, skipping")
        else:
            incomplete.append(feature_set)
    if not incomplete:
        return {"run_id": run_id, "seed": seed, "feature_sets": []}

    config_path, splits_path, input_root = _download_worker_inputs(
        s3, bucket, run_id
    )
    data_config, model_config, train_config = data.load_config(config_path)
    features = list(data_config["features"])
    pipelines, _ = data.prepare_cohort(
        input_root,
        ("UNSORTED",),
        data_config,
        storage={"UNSORTED": "disk"},
        cache_dir=Path("/workspace/block_cache"),
    )
    records = pipelines["UNSORTED"]
    splits = artifacts.load_canonical_splits(splits_path, N_RUNS)
    artifacts.validate_role_membership(
        splits, records=records, dnu_sessions=DNU_SESSIONS
    )
    state = {
        "records": records,
        "splits": splits,
        "features": features,
        "feature_sets": FEATURE_SETS,
        "runs_root": Path("/workspace/output"),
        "model_config": model_config,
        "train_config": train_config,
        "n_spikes": int(data_config["n"]),
        "reuse_baseline": False,
        "manifest_rows": [],
    }
    for feature_set in incomplete:
        run_dir = Path("/workspace/output") / feature_set / f"seed{seed}"
        workflows.run_seed(state, feature_set, seed)
        expected = _completion_identity(run_id, feature_set, seed)
        completion = json.loads((run_dir / "run_complete.json").read_text())
        completion.update(expected)
        io.atomic_json(run_dir / "run_complete.json", completion)
        remote_prefix = _run_prefix(result_prefix, run_id, feature_set, seed)
        for name in upload_order:
            s3.upload_file(
                str(run_dir / name), bucket, f"{remote_prefix}/{name}"
            )

def run_finalizer_worker():
    import pandas as pd
    from ..workflows import ablation as workflows

    install_numpy_pickle_compatibility()
    s3, bucket, run_id, result_prefix = _worker_context()
    n_runs = int(os.environ["ABLATION_N_RUNS"])
    config_path, splits_path, input_root = _download_worker_inputs(
        s3, bucket, run_id
    )
    staging_root = Path("/workspace/finalizer")
    manifest_rows = []
    for feature_set in FEATURE_SET_NAMES:
        for seed in range(n_runs):
            remote_prefix = _run_prefix(
                result_prefix, run_id, feature_set, seed
            )
            expected = _completion_identity(run_id, feature_set, seed)
            if not _remote_run_complete(
                s3, bucket, remote_prefix, expected, TRAINING_ARTIFACTS
            ):
                raise FileNotFoundError(
                    f"Incomplete Batch run: {feature_set}/seed{seed}"
                )
            run_dir = staging_root / "runs" / feature_set / f"seed{seed}"
            for name in ANALYSIS_RUN_ARTIFACTS:
                download(
                    s3,
                    bucket,
                    f"{remote_prefix}/{name}",
                    run_dir / name,
                )
            selected = list(FEATURE_SETS[feature_set])
            manifest_rows.append({
                "pipeline": "UNSORTED",
                "feature_set": feature_set,
                "seed": seed,
                "features": json.dumps(selected),
                "n_features": len(selected),
                "baseline_reused": False,
                "artifact_source": str(run_dir),
            })
    manifest_path = staging_root / "run_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    archive_path = staging_root / "ablation_analysis.zip"

    workflows.finalize(
        manifest_path=manifest_path,
        output_dir=staging_root / "analysis_exports",
        archive_path=archive_path,
        canonical_splits_path=splits_path,
        config_path=config_path,
        unsorted_path=input_root / "unsorted",
        feature_order=FEATURE_SET_NAMES,
        n_runs=n_runs,
    )
    archive_key = os.environ["ABLATION_ARCHIVE_KEY"]
    s3.upload_file(str(archive_path), bucket, archive_key)
    completion = {
        "profile": os.environ["ABLATION_PROFILE"],
        "run_id": run_id,
        "archive_uri": f"s3://{bucket}/{archive_key}",
    }
    put_json(
        s3,
        bucket,
        f"{result_prefix}/control/completions/"
        f"{os.environ['ABLATION_PROFILE'].lower()}-{run_id}.json",
        completion,
    )
    return completion


def run_worker_from_env():
    mode = os.environ["ABLATION_MODE"].upper()
    if mode == "TRAIN":
        return run_training_worker()
    if mode == "FINALIZE":
        return run_finalizer_worker()
    raise ValueError(f"Unknown ablation worker mode: {mode}")
