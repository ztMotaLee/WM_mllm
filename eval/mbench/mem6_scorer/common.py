"""Small dependency-free input, pinning, and receipt helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "c44dbd9dfa0278be728bda2101b0fb96f8547f11"
ENVIRONMENT = (
    "mbencha.environment.spatial_epipolar",
    "mbencha.environment.spatial_reprojection",
    "mbencha.environment.rendering_lighting",
    "mbencha.environment.rendering_style",
)
FAMILIES = {"environment": ENVIRONMENT}


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def upstream_root():
    """Directory holding the importable upstream ``mbench`` package (a checkout on PYTHONPATH)."""
    import importlib.util
    spec = importlib.util.find_spec("mbench")
    if spec is None or spec.origin is None:
        raise RuntimeError("MBench is not importable; clone study-overflow/MBench at "
                           + UPSTREAM_COMMIT + " and add the checkout to PYTHONPATH")
    return Path(spec.origin).resolve().parents[1]


def verify_code():
    lock = json.loads((PACKAGE / "CODE_LOCK.json").read_text())
    if digest({"files": lock["files"], "upstream_files": lock["upstream_files"]}) != lock["tree_sha256"]:
        raise RuntimeError("Scorer code lock is inconsistent")
    upstream = upstream_root()
    for root, files in ((PACKAGE, lock["files"]), (upstream, lock["upstream_files"])):
        for relative, expected in files.items():
            path = root / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError("Scorer code differs from the lock: " + str(path))
    return {"lock": file_ref(PACKAGE / "CODE_LOCK.json"), "tree_sha256": lock["tree_sha256"],
            "upstream_commit": lock["upstream_commit"]}


def read_manifest(path):
    path = Path(path).resolve(strict=True)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else next(
            (payload[key] for key in ("rows", "files", "videos", "items", "exports")
             if isinstance(payload.get(key), list)), None)
        if rows is None:
            raise ValueError("Manifest needs a list or rows/files/videos/items/exports list")
    normalized = []
    seen = set()
    for index, row in enumerate(rows):
        model = row.get("model_id") or row.get("model")
        item = row.get("item_id")
        subset = row.get("subset") or (item.split(":", 1)[0] if item else None)
        ref = row.get("video")
        video = ref.get("path") if isinstance(ref, dict) else ref
        video = row.get("video_path") or row.get("final_path") or video or row.get("path")
        if not all(isinstance(v, str) and v for v in (model, item, subset, video)):
            raise ValueError("Incomplete row " + str(index))
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = path.parent / video_path
        expected = (ref.get("sha256") if isinstance(ref, dict) else None) or row.get(
            "lfs_sha256") or row.get("sha256") or row.get("expected_sha256")
        if expected and expected.startswith("sha256:"):
            expected = expected.removeprefix("sha256:")
        key = (model, item)
        if key in seen:
            raise ValueError("Duplicate model/item in manifest: " + repr(key))
        seen.add(key)
        normalized.append({**row, "model_id": model, "item_id": item, "subset": subset,
                           "video_path": str(video_path.resolve()), "expected_sha256": expected,
                           "manifest_index": index,
                           "condition_id": row.get("condition_id") or row.get("condition")})
    return normalized


def record_id(row):
    return hashlib.sha256((row["model_id"] + "\0" + row["item_id"]).encode()).hexdigest()
