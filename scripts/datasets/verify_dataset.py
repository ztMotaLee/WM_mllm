#!/usr/bin/env python3
"""Verify explicitly listed dataset files without downloading or modifying data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable


MANIFESTS = {
    "compbench-paper-2709": "compbench_paper_2709.sha256",
    "mbench-a-environment-227": "mbench_a_environment_227.sha256",
    "mbench-a-first-frames-547": "mbench_a_first_frames_547.sha256",
}


def relative_path(value: str) -> str:
    """Accept normalized POSIX paths only; never reinterpret absolute paths."""
    if not isinstance(value, str) or not value or any(character in value for character in "\\\x00?#:") or any(ord(character) < 32 for character in value):
        raise ValueError("manifest path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", value) or "://" in value:
        raise ValueError("manifest path must be relative")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("manifest path has an invalid segment")
    return value


def safe_path(root: Path, value: str) -> Path:
    """Resolve an existing regular file, rejecting every child symlink."""
    value = relative_path(value)
    base = root.resolve(strict=True)
    if not base.is_dir():
        raise ValueError("dataset root must be a directory")
    candidate = base
    for part in value.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{value}: symlink is not permitted")
    if not candidate.is_file():
        raise ValueError(f"{value}: missing regular file")
    if not candidate.resolve(strict=True).is_relative_to(base):
        raise ValueError(f"{value}: file escapes dataset root")
    return candidate


def read_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        if len(line) < 67 or line[64:66] != "  " or not re.fullmatch(r"[0-9a-f]{64}", line[:64]):
            raise ValueError(f"checksum manifest line {number}: invalid format")
        name = relative_path(line[66:])
        if name in seen:
            raise ValueError(f"checksum manifest line {number}: duplicate path")
        seen.add(name)
        entries.append((name, line[:64]))
    if not entries:
        raise ValueError("checksum manifest is empty")
    return entries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root: Path, entries: Iterable[tuple[str, str]]) -> dict:
    verified = 0
    total_bytes = 0
    failures: list[dict[str, str]] = []
    for name, expected in entries:
        try:
            path = safe_path(root, name)
            if file_sha256(path) != expected:
                failures.append({"path": name, "reason": "sha256 mismatch"})
            else:
                verified += 1
                total_bytes += path.stat().st_size
        except (OSError, ValueError):
            failures.append({"path": name, "reason": "missing, unreadable, or unsafe file"})
    return {
        "status": "PASS" if not failures else "FAIL",
        "verified_files": verified,
        "verified_bytes": total_bytes,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(MANIFESTS))
    parser.add_argument("--root", required=True, type=Path, help="Root of the downloaded benchmark directory")
    args = parser.parse_args(argv)
    try:
        entries = read_manifest(Path(__file__).parent / MANIFESTS[args.dataset])
        result = verify_files(args.root, entries)
    except (OSError, ValueError):
        print(json.dumps({"status": "FAIL", "reason": "cannot read dataset root or checksum manifest"}))
        return 1
    print(json.dumps({"dataset": args.dataset, "expected_files": len(entries), **result}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
