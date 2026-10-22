from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import io
from ..contract import (
    NULL_PERMUTATIONS,
    NULL_RUN_ID,
    NULL_SPLITS,
    PERMUTATIONS_PER_CHILD,
)
from .common import (
    array_counts,
    build_worker_revision,
    clients,
    download,
    get_json,
    install_numpy_pickle_compatibility,
    preflight_resources,
    objects,
    put_json,
    worker_source_files,
)


RUN_ID = NULL_RUN_ID
INPUT_FILES = (
    "training_config.json",
    "unsorted/AD-CO-DAY150.pkl", "unsorted/CTRL-CO-DAY150.pkl",
    "sorted/AD-CO-DAY150.pkl", "sorted/CTRL-CO-DAY150.pkl",
    "pruned/AD-CO-DAY150.pkl", "pruned/CTRL-CO-DAY150.pkl",
)


def _permutation_count(config):
    count = int(config.get("null_permutations", NULL_PERMUTATIONS))
    if not 1 <= count <= 10_000:
        raise ValueError("null_permutations must be in [1, 10000]")
    return count


def array_size(profile, n_permutations=NULL_PERMUTATIONS):
    profile = profile.upper()
    if profile == "SMOKE":
        return 2
    if profile != "FULL":
        raise ValueError(f"Unknown null profile: {profile}")
    n_permutations = _permutation_count(
        {"null_permutations": n_permutations}
    )
    return NULL_SPLITS * (
        (n_permutations + PERMUTATIONS_PER_CHILD - 1)
        // PERMUTATIONS_PER_CHILD
    )


def shard_seeds(
    profile, index, n_permutations=NULL_PERMUTATIONS
):
    profile, index = profile.upper(), int(index)
    if profile == "SMOKE":
        if index not in (0, 1):
            raise ValueError("SMOKE has two children")
        return [index], [index]
    if profile == "FULL":
        size = array_size(profile, n_permutations)
        if not 0 <= index < size:
            raise ValueError(
                f"FULL child index must be in [0, {size - 1}]"
            )
        start = (index // NULL_SPLITS) * PERMUTATIONS_PER_CHILD
        stop = min(start + PERMUTATIONS_PER_CHILD, n_permutations)
        return list(range(start, stop)), [index % NULL_SPLITS]
    raise ValueError(f"Unknown null profile: {profile}")


def preflight(config):
    checked = preflight_resources(config, INPUT_FILES)
    return {**checked, "inputs": len(checked["inputs"])}


def deploy(config, repo_root):
    repo_root = Path(repo_root)
    bucket = config["bucket"]
    deployment = build_worker_revision(
        config,
        files=worker_source_files(repo_root),
        archive_label="null-worker",
        build_prefix=f"{config['result_prefix']}/control/build",
        tags={"Project": "corrected-null", "ManagedBy": "run_null_batch"},
        source_mode="presigned",
    )
    manifest = {
        "run_id": RUN_ID,
        "image_uri": deployment["image_uri"],
        "job_definition_arn": deployment["job_definition_arn"],
        "source_digest": deployment["source_digest"],
        "build_key": deployment["build_key"],
    }
    s3, = clients(config, "s3")
    put_json(s3, bucket, f"{config['result_prefix']}/control/deployment-v3.json", manifest)
    return manifest


def submit(config, profile):
    s3, batch = clients(config, "s3", "batch")
    deployment = get_json(s3, config["bucket"], f"{config['result_prefix']}/control/deployment-v3.json")
    if profile not in {"SMOKE", "FULL"}:
        raise ValueError(f"Unknown null profile: {profile}")
    n_permutations = _permutation_count(config)
    size = array_size(profile, n_permutations)
    environment = [
        {"name": "NULL_PROFILE", "value": profile}, {"name": "NULL_RUN_ID", "value": RUN_ID},
        {"name": "NULL_PERMUTATIONS", "value": str(n_permutations)},
        {"name": "NULL_BUCKET", "value": config["bucket"]},
        {"name": "NULL_INPUT_PREFIX", "value": config["input_prefix"]},
        {"name": "NULL_RESULT_PREFIX", "value": config["result_prefix"]},
    ]
    response = batch.submit_job(
        jobName=f"null-v3-{profile.lower()}-{int(time.time())}",
        jobQueue=config["job_queue"], jobDefinition=deployment["job_definition_arn"],
        arrayProperties={"size": size}, containerOverrides={"environment": environment},
    )
    submission = {
        "job_id": response["jobId"], "profile": profile,
        "array_size": size, "n_permutations": n_permutations,
        "run_id": RUN_ID,
    }
    put_json(s3, config["bucket"], f"{config['result_prefix']}/control/latest_submission-v3.json", submission)
    return submission


def monitor(config, submission=None, wait=True, poll_seconds=30):
    s3, batch = clients(config, "s3", "batch")
    submission = submission or get_json(s3, config["bucket"], f"{config['result_prefix']}/control/latest_submission-v3.json")
    while True:
        counts = array_counts(batch, submission["job_id"])
        terminal = counts["SUCCEEDED"] + counts["FAILED"] == submission["array_size"]
        snapshot = {**submission, "counts": counts, "terminal": terminal}
        if terminal or not wait:
            if counts["FAILED"]:
                raise RuntimeError(f"{counts['FAILED']} null array children failed")
            return snapshot
        time.sleep(poll_seconds)


def consolidate_batch(config, local_root):
    from ..workflows import null as workflows
    s3, = clients(config, "s3")
    bucket = config["bucket"]
    local_root = Path(local_root); shutil.rmtree(local_root, ignore_errors=True); local_root.mkdir(parents=True)
    prefix = f"{config['result_prefix']}/shards/{RUN_ID}/"
    downloads = []
    for obj in objects(s3, bucket, prefix):
        parts = Path(obj["Key"][len(prefix):]).parts
        if len(parts) >= 2:
            relative = Path(*parts[1:])
            if relative.parts[0] == "progress" or parts[0] == "000":
                downloads.append((obj["Key"], local_root / relative))
    with ThreadPoolExecutor(max_workers=64) as pool:
        list(pool.map(lambda item: download(s3, bucket, *item), downloads))
    results = workflows.consolidate(
        local_root,
        permutation_seeds=range(_permutation_count(config)),
        split_seeds=range(NULL_SPLITS),
    )
    archive = local_root / "null_distribution.zip"
    io.write_zip(sorted(local_root.glob("*.csv*")) + [local_root / name for name in ("null_summary.json", "run_manifest.json") if (local_root / name).exists()], archive)
    final_prefix = f"{config['result_prefix']}/final/{RUN_ID}"
    for path in local_root.rglob("*"):
        if path.is_file():
            s3.upload_file(str(path), bucket, f"{final_prefix}/{path.relative_to(local_root).as_posix()}")
    return {**results, "archive_path": archive}


def run_action(action, config, *, repo_root, wait=True):
    action = action.upper()
    if action == "PREFLIGHT":
        return preflight(config)
    if action == "DEPLOY":
        return deploy(config, repo_root)
    if action in {"SMOKE", "FULL"}:
        return monitor(config, submit(config, action), wait=wait)
    if action == "MONITOR":
        return monitor(config, wait=wait)
    if action == "CONSOLIDATE":
        return consolidate_batch(config, Path(repo_root) / "null_distribution" / "null_batch_state" / RUN_ID)
    raise ValueError(f"Unknown Batch action: {action}")


def run_worker_from_env():
    from ..workflows import null as workflows
    region, bucket = os.environ.get("AWS_REGION", "eu-west-1"), os.environ["NULL_BUCKET"]
    input_prefix, result_prefix = os.environ["NULL_INPUT_PREFIX"], os.environ["NULL_RESULT_PREFIX"]
    profile = os.environ["NULL_PROFILE"]
    index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", 0))
    n_permutations = int(
        os.environ.get("NULL_PERMUTATIONS", NULL_PERMUTATIONS)
    )
    permutations, splits = shard_seeds(
        profile, index, n_permutations
    )
    s3, = clients({"region": region}, "s3")
    input_root = Path("/cache/input")
    for relative in INPUT_FILES:
        download(s3, bucket, f"{input_prefix}/{relative}", input_root / relative)
    output_root = Path("/workspace/output") / RUN_ID
    shard_prefix = f"{result_prefix}/shards/{RUN_ID}/{index:03d}"
    for obj in objects(s3, bucket, shard_prefix + "/"):
        download(s3, bucket, obj["Key"], output_root / obj["Key"][len(shard_prefix) + 1:])
    install_numpy_pickle_compatibility()
    setup_permutations = (
        range(2) if profile == "SMOKE" else range(n_permutations)
    )
    setup_splits = range(2) if profile == "SMOKE" else range(NULL_SPLITS)
    state = workflows.setup(
        input_root,
        output_root,
        permutation_seeds=setup_permutations,
        split_seeds=setup_splits,
        cache_dir="/workspace/block_cache",
        config_path=input_root / "training_config.json",
    )

    def upload_fold(directory, _perm_seed, _split_seed):
        for path in directory.iterdir():
            s3.upload_file(str(path), bucket, f"{shard_prefix}/{path.relative_to(output_root).as_posix()}")

    try:
        workflows.run_folds(
            state, permutation_seeds=permutations, split_seeds=splits,
            on_fold_complete=upload_fold,
        )
        for name in ("run_manifest.json", "split_manifest.csv", "global_permutation_maps.csv"):
            s3.upload_file(str(output_root / name), bucket, f"{shard_prefix}/{name}")
    finally:
        workflows.cleanup(state)
