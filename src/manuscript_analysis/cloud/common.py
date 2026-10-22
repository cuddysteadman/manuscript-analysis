from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import boto3

from .. import io


_DEPLOY_ENV_KEYS = {
    "region": "AWS_REGION",
    "account_id": "AWS_ACCOUNT_ID",
    "bucket": "AWS_S3_BUCKET",
    "input_prefix": "MANUSCRIPT_INPUT_PREFIX",
    "job_queue": "AWS_BATCH_JOB_QUEUE",
    "ecr_repository": "AWS_ECR_REPOSITORY",
    "codebuild_project": "AWS_CODEBUILD_PROJECT",
    "job_definition": "AWS_BATCH_JOB_DEFINITION",
}


def _read_env_file(path):
    values = {}
    for line_number, raw_line in enumerate(
        Path(path).read_text().splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(
                f"{path}:{line_number}: expected NAME=value"
            )
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not name:
            raise ValueError(f"{path}:{line_number}: empty variable name")
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[name] = value
    return values


def load_deploy_config(workflow, env_file=None):
    workflow = workflow.lower()
    workflow_keys = {
        "ablation": {
            "result_prefix": "ABLATION_RESULT_PREFIX",
        },
        "null": {
            "result_prefix": "NULL_RESULT_PREFIX",
            "null_permutations": "NULL_PERMUTATIONS",
        },
    }
    if workflow not in workflow_keys:
        raise ValueError("workflow must be 'ablation' or 'null'")
    path = (
        Path(env_file)
        if env_file is not None
        else Path(__file__).resolve().parents[3] / "deploy.local.env"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local deployment config: {path}. "
            "Create deploy.local.env from deploy.local.env.example."
        )
    values = {**_read_env_file(path), **os.environ}
    keys = {**_DEPLOY_ENV_KEYS, **workflow_keys[workflow]}
    missing = [
        environment_name
        for environment_name in keys.values()
        if not values.get(environment_name)
    ]
    if missing:
        raise RuntimeError(
            "Missing deployment environment variables: "
            + ", ".join(sorted(missing))
        )
    config = {
        config_name: values[environment_name]
        for config_name, environment_name in keys.items()
    }
    if workflow == "null":
        config["null_permutations"] = int(config["null_permutations"])
    return config


def clients(config, *names):
    return tuple(
        boto3.client(name, region_name=config["region"]) for name in names
    )


def preflight_resources(
    config, input_files, *, codebuild=False, require_enabled=False
):
    s3, sts, batch, ecr = clients(config, "s3", "sts", "batch", "ecr")
    identity = sts.get_caller_identity()
    if config.get("account_id") and identity["Account"] != str(
        config["account_id"]
    ):
        raise RuntimeError(f"Unexpected AWS account: {identity['Account']}")
    inputs = []
    for relative in input_files:
        key = f"{config['input_prefix']}/{relative}"
        head = s3.head_object(Bucket=config["bucket"], Key=key)
        inputs.append({
            "key": key,
            "etag": head["ETag"],
            "size": int(head["ContentLength"]),
            "version_id": head.get("VersionId"),
        })
    queues = batch.describe_job_queues(
        jobQueues=[config["job_queue"]]
    )["jobQueues"]
    definitions = batch.describe_job_definitions(
        jobDefinitionName=config["job_definition"], status="ACTIVE"
    )["jobDefinitions"]
    repositories = ecr.describe_repositories(
        repositoryNames=[config["ecr_repository"]]
    )["repositories"]
    projects = (
        clients(config, "codebuild")[0].batch_get_projects(
            names=[config["codebuild_project"]]
        )["projects"]
        if codebuild else []
    )
    if (
        not queues
        or require_enabled and queues[0]["state"] != "ENABLED"
        or not definitions or not repositories or codebuild and not projects
    ):
        raise RuntimeError("Configured Batch resources are unavailable")
    result = {
        "identity": identity["Arn"],
        "inputs": inputs,
        "job_queue_arn": queues[0]["jobQueueArn"],
        "job_definition_arn": max(
            definitions, key=lambda row: row["revision"]
        )["jobDefinitionArn"],
        "worker_repository_arn": repositories[0]["repositoryArn"],
    }
    if codebuild:
        result["codebuild_project_arn"] = projects[0]["arn"]
    return result


def worker_source_files(repo_root):
    root = Path(repo_root)
    paths = []
    for source in sorted((root / "src" / "manuscript_analysis").rglob("*.py")):
        paths.append((source, source.relative_to(root).as_posix()))
    paths.extend([
        (root / "config" / "training_config.json",
         "config/training_config.json"),
        (root / "pyproject.toml", "pyproject.toml"),
        (root / "infra" / "batch" / "Dockerfile", "Dockerfile"),
        (root / "infra" / "batch" / "requirements.lock", "requirements.lock"),
        (root / "infra" / "batch" / "batch_entrypoint.py",
         "batch_entrypoint.py"),
        (root / "infra" / "batch" / "buildspec.yml", "buildspec.yml"),
    ])
    for policy in sorted((root / "infra" / "batch" / "policies").glob("*.json")):
        paths.append((policy, f"policies/{policy.name}"))
    missing = [str(source) for source, _ in paths if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Batch build sources: {missing}")
    return tuple(paths)


def install_numpy_pickle_compatibility():
    for serialized, runtime in {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }.items():
        sys.modules.setdefault(serialized, importlib.import_module(runtime))


def put_json(s3, bucket, key, payload):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=io.json_bytes(payload),
        ContentType="application/json",
    )


def get_json(s3, bucket, key):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def objects(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents", [])


def download(s3, bucket, key, destination):
    destination = Path(destination)
    head = s3.head_object(Bucket=bucket, Key=key)
    if destination.exists() and destination.stat().st_size == head["ContentLength"]:
        return destination
    with io.atomic_target(destination) as temporary:
        s3.download_file(bucket, key, str(temporary))
        if temporary.stat().st_size != head["ContentLength"]:
            raise IOError(f"Short S3 download: {key}")
    return destination


def source_digest(files):
    digest = hashlib.sha256()
    for source, archive_name in sorted(
        ((Path(source), archive_name) for source, archive_name in files),
        key=lambda item: item[1],
    ):
        digest.update(archive_name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def build_worker_revision(
    config,
    *,
    files,
    archive_label,
    build_prefix,
    tags=None,
    source_mode="s3",
):
    bucket = config["bucket"]
    s3, ecr, codebuild, batch = clients(config, "s3", "ecr", "codebuild", "batch")
    digest = source_digest(files)
    archive = Path(tempfile.gettempdir()) / f"{archive_label}-{digest}.zip"
    io.write_zip(files, archive)
    build_key = f"{build_prefix}/{archive.name}"
    s3.upload_file(str(archive), bucket, build_key)

    repository = ecr.describe_repositories(
        repositoryNames=[config["ecr_repository"]]
    )["repositories"][0]
    image_uri = f"{repository['repositoryUri']}:{digest}"
    try:
        ecr.describe_images(
            repositoryName=config["ecr_repository"],
            imageIds=[{"imageTag": digest}],
        )
    except ecr.exceptions.ImageNotFoundException:
        request = {
            "projectName": config["codebuild_project"],
            "environmentVariablesOverride": [
                {"name": "IMAGE_URI", "value": image_uri, "type": "PLAINTEXT"}
            ],
        }
        if source_mode == "presigned":
            source_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": build_key},
                ExpiresIn=3600,
            )
            request.update({
                "sourceTypeOverride": "NO_SOURCE",
                "buildspecOverride": """version: 0.2
phases:
  pre_build:
    commands:
      - curl --fail --silent --show-error "$SOURCE_URL" -o /tmp/worker.zip
      - unzip -q /tmp/worker.zip -d "$CODEBUILD_SRC_DIR"
      - ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
      - REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
      - aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  build:
    commands:
      - docker build --pull -t "$IMAGE_URI" "$CODEBUILD_SRC_DIR"
  post_build:
    commands:
      - docker push "$IMAGE_URI"
""",
            })
            request["environmentVariablesOverride"].append(
                {"name": "SOURCE_URL", "value": source_url, "type": "PLAINTEXT"}
            )
        elif source_mode == "s3":
            request["sourceLocationOverride"] = f"{bucket}/{build_key}"
        else:
            raise ValueError("source_mode must be 's3' or 'presigned'")
        build = codebuild.start_build(**request)["build"]
        while True:
            current = codebuild.batch_get_builds(ids=[build["id"]])["builds"][0]
            status = current["buildStatus"]
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "FAULT", "STOPPED", "TIMED_OUT"}:
                messages = [
                    context.get("message", "")
                    for phase in current.get("phases", [])
                    for context in phase.get("contexts", [])
                    if context.get("message")
                ]
                detail = f": {'; '.join(messages)}" if messages else ""
                raise RuntimeError(f"CodeBuild failed: {status}{detail}")
            time.sleep(15)

    base = max(
        batch.describe_job_definitions(
            jobDefinitionName=config["job_definition"], status="ACTIVE"
        )["jobDefinitions"],
        key=lambda row: row["revision"],
    )
    properties = dict(base["containerProperties"])
    properties["image"] = image_uri
    request = {
        "jobDefinitionName": config["job_definition"],
        "type": "container",
        "containerProperties": properties,
    }
    for key in (
        "parameters", "retryStrategy", "timeout", "platformCapabilities",
        "propagateTags",
    ):
        if key in base:
            request[key] = base[key]
    request["tags"] = tags or base.get("tags", {})
    definition = batch.register_job_definition(**request)
    return {
        "source_digest": digest,
        "image_uri": image_uri,
        "job_definition_arn": definition["jobDefinitionArn"],
        "build_key": build_key,
    }


def array_counts(batch, job_id):
    counts = {}
    for status in (
        "SUBMITTED", "PENDING", "RUNNABLE", "STARTING",
        "RUNNING", "SUCCEEDED", "FAILED",
    ):
        pages = batch.get_paginator("list_jobs").paginate(
            arrayJobId=job_id, jobStatus=status
        )
        counts[status] = sum(
            len(page.get("jobSummaryList", [])) for page in pages
        )
    return counts
