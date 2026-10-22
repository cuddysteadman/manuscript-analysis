from __future__ import annotations

import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np


_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_UNIX = 3


@contextmanager
def atomic_target(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value):
    return value.item() if isinstance(value, np.generic) else str(value)


def json_bytes(payload):
    return json.dumps(
        payload, indent=2, sort_keys=True, default=_json_default
    ).encode()


def atomic_json(path, payload):
    with atomic_target(path) as temporary:
        temporary.write_bytes(json_bytes(payload))


def atomic_csv(frame, path):
    compression = (
        {"method": "gzip", "mtime": 0} if Path(path).suffix == ".gz" else None
    )
    with atomic_target(path) as temporary:
        frame.to_csv(
            temporary, index=False, compression=compression, lineterminator="\n"
        )


def write_zip(entries, archive_path):
    members = sorted(
        (
            entry if isinstance(entry, tuple) else (entry, Path(entry).name)
            for entry in entries
        ),
        key=lambda member: member[1],
    )
    with atomic_target(archive_path) as temporary, zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED
    ) as output:
        for source, archive_name in members:
            info = zipfile.ZipInfo(archive_name, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = _ZIP_UNIX
            info.external_attr = 0o100644 << 16
            output.writestr(info, Path(source).read_bytes())
